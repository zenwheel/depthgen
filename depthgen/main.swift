//
//  main.swift
//  depthgen
//
//  Created by Scott Jann on 12/24/25.
//

import Foundation
import CoreML
import Vision
import AppKit
import CoreImage

// MARK: - Entry point (top-level code, no @main)

let args = CommandLine.arguments.dropFirst()
guard !args.isEmpty else {
	fputs("Usage: depthpro <image1> [image2 ...]\n", stderr)
	exit(1)
}

do {
	let runner = try DepthProRunner()
	
	for path in args {
		let url = URL(fileURLWithPath: path)
		guard let image = NSImage(contentsOf: url) else {
			fputs("Failed to load image: \(path)\n", stderr)
			continue
		}
		
		fputs("Processing \(path)...\n", stderr)
		
		let originalSize = image.size
		
		let semaphore = DispatchSemaphore(value: 0)
		var resultCG: CGImage?
		
		runner.depthGrayscaleResized(from: image,
									 targetSize: originalSize) { cgImage in
			resultCG = cgImage
			semaphore.signal()
		}
		
		semaphore.wait()
		
		guard let outCGImage = resultCG else {
			fputs("Depth estimation failed for \(path)\n", stderr)
			continue
		}
		
		let outURL = outputURL(for: url)
		if savePNG(cgImage: outCGImage, to: outURL, size: originalSize) {
			fputs("Saved \(outURL.path)\n", stderr)
		} else {
			fputs("Failed to save: \(outURL.path)\n", stderr)
		}
	}
} catch {
	fputs("Error initializing Depth Pro: \(error)\n", stderr)
	exit(1)
}

// MARK: - Helpers

func outputURL(for input: URL) -> URL {
	let base = input.deletingPathExtension().lastPathComponent
	let dir = input.deletingLastPathComponent()
	let outName = "\(base)-depth.png"
	return dir.appendingPathComponent(outName)
}

func savePNG(cgImage: CGImage, to url: URL, size: NSSize) -> Bool {
	let rep = NSBitmapImageRep(cgImage: cgImage)
	rep.size = size
	
	guard let data = rep.representation(using: .png, properties: [:]) else {
		return false
	}
	
	do {
		try data.write(to: url)
		return true
	} catch {
		return false
	}
}

// MARK: - Depth Pro runner

final class DepthProRunner {
	private let model: VNCoreMLModel
	private let ciContext = CIContext(options: nil)
	
	init() throws {
		// Replace with your actual generated model class.
		let coreMLModel = try DepthProNormalizedInverseDepthPruned10QuantizedLinear(
			configuration: MLModelConfiguration()
		).model
		self.model = try VNCoreMLModel(for: coreMLModel)
	}
	
	/// Run Depth Pro, get grayscale depth resized back to targetSize.
	func depthGrayscaleResized(from inputImage: NSImage,
							   targetSize: NSSize,
							   completion: @escaping (CGImage?) -> Void) {
		guard let tiffData = inputImage.tiffRepresentation,
			  let ciImage = CIImage(data: tiffData) else {
			completion(nil)
			return
		}
		
		let request = VNCoreMLRequest(model: model) { request, error in
			guard error == nil,
				  let results = request.results as? [VNPixelBufferObservation],
				  let obs = results.first else {
				completion(nil)
				return
			}
			
			let depthPixelBuffer = obs.pixelBuffer
			
			// Convert model depth output to grayscale CGImage at model resolution.
			guard let depthGrayCG = self.pixelBufferToGrayscale(depthPixelBuffer) else {
				completion(nil)
				return
			}
			
			// Resize grayscale depth to original size.
			let resized = self.resizeCGImage(depthGrayCG,
											 to: targetSize,
											 originalOrientationOf: ciImage)
			completion(resized)
		}
		
		// Let Vision/Core ML handle resizing/cropping to the model's expected (1536x1536) input.
		request.imageCropAndScaleOption = .scaleFill
		
		let handler = VNImageRequestHandler(ciImage: ciImage, options: [:])
		DispatchQueue.global(qos: .userInitiated).async {
			do {
				try handler.perform([request])
			} catch {
				completion(nil)
			}
		}
	}
	
	/// Convert Depth Pro float32 output to 8-bit grayscale CGImage.
	private func pixelBufferToGrayscale(_ pixelBuffer: CVPixelBuffer) -> CGImage? {
#if arch(arm64)
		CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
		defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
		
		let width  = CVPixelBufferGetWidth(pixelBuffer)
		let height = CVPixelBufferGetHeight(pixelBuffer)
		let dataSize = CVPixelBufferGetDataSize(pixelBuffer)
		
		print("DEBUG: pixelBuffer w=\(width) h=\(height) dataSize=\(dataSize)")
		
		guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
			print("DEBUG: no baseAddress")
			return nil
		}
		
		// Float16: 2 bytes per float
		let requiredFloat16Count = width * height
		let requiredBytes = requiredFloat16Count * MemoryLayout<Float16>.stride  // 2 bytes
		
		print("DEBUG: required floats=\(requiredFloat16Count) bytes=\(requiredBytes)")
		
		guard dataSize >= requiredBytes else {
			print("DEBUG: dataSize \(dataSize) < required \(requiredBytes)")
			return nil
		}
		
		let float16Ptr = baseAddress.assumingMemoryBound(to: Float16.self)
		let float16s = UnsafeBufferPointer(start: float16Ptr, count: requiredFloat16Count)
		
		// Convert Float16 -> Float32 for min/max calculation
		var minVal = Float.greatestFiniteMagnitude
		var maxVal = -Float.greatestFiniteMagnitude
		for v16 in float16s {
			let v = Float(v16)
			if v.isNaN { continue }
			minVal = min(minVal, v)
			maxVal = max(maxVal, v)
		}
		
		print("DEBUG: minVal=\(minVal) maxVal=\(maxVal)")
		
		if maxVal <= minVal || !minVal.isFinite || !maxVal.isFinite {
			print("DEBUG: bad min/max values")
			return nil
		}
		
		// Normalize and create 8-bit grayscale
		var grayBytes = [UInt8](repeating: 0, count: requiredFloat16Count)
		for i in 0..<requiredFloat16Count {
			let v16 = float16s[i]
			var v = Float(v16)
			if v.isNaN { v = minVal }
			let n = (v - minVal) / (maxVal - minVal)
			grayBytes[i] = UInt8(clamping: Int(n * 255.0))
		}
		
		let colorSpace = CGColorSpaceCreateDeviceGray()
		guard let provider = CGDataProvider(data: Data(grayBytes) as CFData) else {
			print("DEBUG: CGDataProvider failed")
			return nil
		}
		
		let cgImage = CGImage(
			width: width,
			height: height,
			bitsPerComponent: 8,
			bitsPerPixel: 8,
			bytesPerRow: width,
			space: colorSpace,
			bitmapInfo: CGBitmapInfo(rawValue: 0),
			provider: provider,
			decode: nil,
			shouldInterpolate: false,
			intent: .defaultIntent
		)
		
		return cgImage
#else
		return nil
#endif
	}

	/// Resize a CGImage to targetSize using Core Image.
	private func resizeCGImage(_ image: CGImage,
							   to targetSize: NSSize,
							   originalOrientationOf ciImage: CIImage) -> CGImage? {
		let ciInput = CIImage(cgImage: image).oriented(ciImage.orientation)
		
		let scaleX = targetSize.width / CGFloat(image.width)
		let scaleY = targetSize.height / CGFloat(image.height)
		
		let transform = CGAffineTransform(scaleX: scaleX, y: scaleY)
		let ciOutput = ciInput.transformed(by: transform)
		
		let rect = CGRect(origin: .zero, size: targetSize)
		
		return ciContext.createCGImage(ciOutput, from: rect)
	}
}

// Small helper so we can preserve orientation when resizing.
private extension CIImage {
	var orientation: CGImagePropertyOrientation {
		if let val = properties[kCGImagePropertyOrientation as String] as? UInt32,
		   let ori = CGImagePropertyOrientation(rawValue: val) {
			return ori
		}
		return .up
	}
}
