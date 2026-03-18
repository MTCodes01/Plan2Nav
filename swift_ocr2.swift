import Vision
import AppKit

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    print("Usage: swift ocr.swift <image_path>")
    exit(1)
}

let path = arguments[1]
print("Loading path: \(path)")

let fileManager = FileManager.default
if !fileManager.fileExists(atPath: path) {
    print("FILE_NOT_FOUND")
    exit(1)
}

guard let image = NSImage(contentsOfFile: path),
      let cgImage = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("FAILED_TO_LOAD_IMAGE_OR_CG")
    exit(1)
}

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
let request = VNRecognizeTextRequest { request, error in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    for obs in observations {
        let text = obs.topCandidates(1).first?.string ?? ""
        let box = obs.boundingBox
        print("\(text)|\(box.origin.x)|\(box.origin.y)|\(box.size.width)|\(box.size.height)")
    }
}
request.recognitionLevel = .accurate

do {
    try handler.perform([request])
} catch {
    print("FAILED_OCR")
}
