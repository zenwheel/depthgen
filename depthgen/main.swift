//
//  main.swift
//  depthgen
//
//  Created by Scott Jann on 12/24/25.
//

import Foundation
import CoreML
import Vision
import ImageIO
import Accelerate
import UniformTypeIdentifiers

// MARK: - Entry point (top-level code, no @main)

let args = CommandLine.arguments.dropFirst()
guard !args.isEmpty else {
	fputs("Usage: depthgen <image1> [image2 ...]\n", stderr)
	exit(1)
}

let runner: DepthProRunner
do {
	runner = try DepthProRunner()
} catch {
	fputs("Error initializing Depth Pro: \(error)\n", stderr)
	exit(1)
}

var anyFailed = false
for path in args {
	let ok = autoreleasepool { process(path: path, runner: runner) }
	if !ok {
		anyFailed = true
	}
}
exit(anyFailed ? 1 : 0)

// MARK: - Helpers

func process(path: String, runner: DepthProRunner) -> Bool {
	let url = URL(fileURLWithPath: path)
	guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
		  let image = CGImageSourceCreateImageAtIndex(source, 0, nil),
		  image.width > 0, image.height > 0 else {
		fputs("Failed to load image: \(path)\n", stderr)
		return false
	}

	// EXIF orientation is metadata; the decoded pixels are unrotated. Vision is
	// told the orientation below, so its output (and ours) is in display space.
	var orientation = CGImagePropertyOrientation.up
	if let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
	   let raw = (props[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value,
	   let ori = CGImagePropertyOrientation(rawValue: raw) {
		orientation = ori
	}

	let swapsDimensions: Set<CGImagePropertyOrientation> = [.left, .leftMirrored, .right, .rightMirrored]
	let outWidth = swapsDimensions.contains(orientation) ? image.height : image.width
	let outHeight = swapsDimensions.contains(orientation) ? image.width : image.height

	fputs("Processing \(path) (\(outWidth)x\(outHeight))...\n", stderr)

	guard let depthImage = runner.depthMap(from: image, orientation: orientation,
										   width: outWidth, height: outHeight) else {
		fputs("Depth estimation failed for \(path)\n", stderr)
		return false
	}

	let outURL = outputURL(for: url)
	guard savePNG(cgImage: depthImage, to: outURL) else {
		fputs("Failed to save: \(outURL.path)\n", stderr)
		return false
	}

	fputs("Saved \(outURL.path)\n", stderr)
	return true
}

func outputURL(for input: URL) -> URL {
	let base = input.deletingPathExtension().lastPathComponent
	let dir = input.deletingLastPathComponent()
	let outName = "\(base)-depth.png"
	return dir.appendingPathComponent(outName)
}

func savePNG(cgImage: CGImage, to url: URL) -> Bool {
	guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
		return false
	}
	CGImageDestinationAddImage(destination, cgImage, nil)
	return CGImageDestinationFinalize(destination)
}

// MARK: - Depth Pro runner

final class DepthProRunner {
	private let model: VNCoreMLModel

	init() throws {
		let coreMLModel = try DepthProNormalizedInverseDepthPruned10QuantizedLinear(
			configuration: MLModelConfiguration()
		).model
		self.model = try VNCoreMLModel(for: coreMLModel)
	}

	/// Run Depth Pro, get grayscale depth resized back to width x height (display orientation).
	func depthMap(from image: CGImage,
				  orientation: CGImagePropertyOrientation,
				  width: Int, height: Int) -> CGImage? {
		let request = VNCoreMLRequest(model: model)
		// Let Vision/Core ML handle resizing to the model's expected (1536x1536) input;
		// the output is stretched back to the original size below.
		request.imageCropAndScaleOption = .scaleFill

		let handler = VNImageRequestHandler(cgImage: image, orientation: orientation, options: [:])
		do {
			try handler.perform([request])
		} catch {
			fputs("Vision request failed: \(error)\n", stderr)
			return nil
		}

		guard let observation = (request.results as? [VNPixelBufferObservation])?.first,
			  let grayscale = normalizedGrayscale(from: observation.pixelBuffer) else {
			return nil
		}

		return resize(grayscale, width: width, height: height)
	}

	/// Convert Depth Pro float output to 8-bit grayscale, normalized to the full 0-255 range.
	private func normalizedGrayscale(from pixelBuffer: CVPixelBuffer) -> CGImage? {
		CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
		defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

		let width = CVPixelBufferGetWidth(pixelBuffer)
		let height = CVPixelBufferGetHeight(pixelBuffer)
		guard width > 0, height > 0,
			  let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
			return nil
		}

		var source = vImage_Buffer(data: baseAddress,
								   height: vImagePixelCount(height),
								   width: vImagePixelCount(width),
								   rowBytes: CVPixelBufferGetBytesPerRow(pixelBuffer))

		let count = width * height
		var floats = [Float](repeating: 0, count: count)
		let format = CVPixelBufferGetPixelFormatType(pixelBuffer)
		let converted = floats.withUnsafeMutableBufferPointer { buffer -> Bool in
			var destination = vImage_Buffer(data: buffer.baseAddress,
											height: vImagePixelCount(height),
											width: vImagePixelCount(width),
											rowBytes: width * MemoryLayout<Float>.stride)
			switch format {
			case kCVPixelFormatType_OneComponent16Half,
				 kCVPixelFormatType_DepthFloat16,
				 kCVPixelFormatType_DisparityFloat16:
				return vImageConvert_Planar16FtoPlanarF(&source, &destination, vImage_Flags(kvImageNoFlags)) == kvImageNoError
			case kCVPixelFormatType_OneComponent32Float,
				 kCVPixelFormatType_DepthFloat32,
				 kCVPixelFormatType_DisparityFloat32:
				return vImageCopyBuffer(&source, &destination, MemoryLayout<Float>.stride, vImage_Flags(kvImageNoFlags)) == kvImageNoError
			default:
				fputs("Unsupported depth output pixel format: \(format)\n", stderr)
				return false
			}
		}
		guard converted else {
			return nil
		}

		var minVal: Float = 0
		var maxVal: Float = 0
		// NaN/inf propagate through a sum, so a finite sum means every value is
		// finite and the vectorized min/max can be trusted.
		var sum: Float = 0
		vDSP_sve(floats, 1, &sum, vDSP_Length(count))
		if sum.isFinite {
			vDSP_minv(floats, 1, &minVal, vDSP_Length(count))
			vDSP_maxv(floats, 1, &maxVal, vDSP_Length(count))
		} else {
			minVal = .greatestFiniteMagnitude
			maxVal = -.greatestFiniteMagnitude
			for value in floats where value.isFinite {
				minVal = min(minVal, value)
				maxVal = max(maxVal, value)
			}
			guard minVal.isFinite, maxVal.isFinite else {
				return nil
			}
			for i in 0..<count where !floats[i].isFinite {
				floats[i] = floats[i] == .infinity ? maxVal : minVal
			}
		}
		guard maxVal > minVal else {
			return nil
		}

		var grayBytes = [UInt8](repeating: 0, count: count)
		let quantized = floats.withUnsafeMutableBufferPointer { input -> Bool in
			grayBytes.withUnsafeMutableBufferPointer { output -> Bool in
				var sourceBuffer = vImage_Buffer(data: input.baseAddress,
												 height: vImagePixelCount(height),
												 width: vImagePixelCount(width),
												 rowBytes: width * MemoryLayout<Float>.stride)
				var destinationBuffer = vImage_Buffer(data: output.baseAddress,
													  height: vImagePixelCount(height),
													  width: vImagePixelCount(width),
													  rowBytes: width)
				return vImageConvert_PlanarFtoPlanar8(&sourceBuffer, &destinationBuffer, maxVal, minVal, vImage_Flags(kvImageNoFlags)) == kvImageNoError
			}
		}
		guard quantized else {
			return nil
		}

		guard let provider = CGDataProvider(data: Data(grayBytes) as CFData) else {
			return nil
		}

		return CGImage(
			width: width,
			height: height,
			bitsPerComponent: 8,
			bitsPerPixel: 8,
			bytesPerRow: width,
			space: CGColorSpaceCreateDeviceGray(),
			bitmapInfo: CGBitmapInfo(rawValue: 0),
			provider: provider,
			decode: nil,
			shouldInterpolate: false,
			intent: .defaultIntent
		)
	}

	/// Resize a grayscale CGImage to the target pixel size.
	private func resize(_ image: CGImage, width: Int, height: Int) -> CGImage? {
		if image.width == width && image.height == height {
			return image
		}
		guard let context = CGContext(data: nil,
									  width: width,
									  height: height,
									  bitsPerComponent: 8,
									  bytesPerRow: 0,
									  space: CGColorSpaceCreateDeviceGray(),
									  bitmapInfo: CGImageAlphaInfo.none.rawValue) else {
			return nil
		}
		context.interpolationQuality = .high
		context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
		return context.makeImage()
	}
}
