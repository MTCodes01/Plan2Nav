import Vision
import AppKit

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print("Usage: swift ocr.swift <image_path>")
    exit(1)
}

let path = arguments[1]
let url = URL(fileURLWithPath: path)

guard let image = NSImage(contentsOf: url),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("FAILED_TO_LOAD")
    exit(1)
}

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
let request = VNRecognizeTextRequest { request, error in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for obs in observations {
        if let cand = obs.topCandidates(1).first {
             print("\(cand.string)|\(obs.boundingBox.origin.x)|\(obs.boundingBox.origin.y)|\(obs.boundingBox.size.width)|\(obs.boundingBox.size.height)")
        }
    }
}
// Set recognition level to accurate
request.recognitionLevel = .accurate

do {
    try handler.perform([request])
} catch {
    print("FAILED_OCR")
}
