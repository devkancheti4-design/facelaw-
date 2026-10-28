// grab: capture frames from a camera with AVFoundation and save them as PNG.
//   grab <device-substring> <out-dir> <count> <interval-seconds>
// Prints the camera authorisation status first, so a permission problem is visible, not a silent hang.
import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
let want = args.count > 1 ? args[1] : "MacBook Pro Camera"
let outDir = args.count > 2 ? args[2] : "."
let count = args.count > 3 ? Int(args[3]) ?? 1 : 1
let interval = args.count > 4 ? Double(args[4]) ?? 2.0 : 2.0

// everything printed also goes to <out-dir>/grab.log (an app launched with `open` has no terminal)
let logURL = URL(fileURLWithPath: outDir).appendingPathComponent("grab.log")
FileManager.default.createFile(atPath: logURL.path, contents: nil)
let logH = try? FileHandle(forWritingTo: logURL)
func print(_ items: Any...) {
    let line = items.map { "\($0)" }.joined(separator: " ") + "\n"
    FileHandle.standardOutput.write(line.data(using: .utf8)!)
    logH?.write(line.data(using: .utf8)!)
}
let status = AVCaptureDevice.authorizationStatus(for: .video)
print("camera authorisation:", ["notDetermined", "restricted", "denied", "authorized"][status.rawValue])
if status == .notDetermined {
    let sem = DispatchSemaphore(value: 0)
    AVCaptureDevice.requestAccess(for: .video) { ok in print("access granted:", ok); sem.signal() }
    sem.wait()
}
let found = AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
                                             mediaType: .video, position: .unspecified).devices
print("cameras:", found.map { $0.localizedName })
guard let dev = found.first(where: { $0.localizedName.contains(want) }) else { print("no camera matching", want); exit(1) }

final class Grabber: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    var latest: CVPixelBuffer?
    let lock = NSLock()
    func captureOutput(_ o: AVCaptureOutput, didOutput sb: CMSampleBuffer, from c: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        lock.lock(); latest = pb; lock.unlock()
    }
}
let session = AVCaptureSession()
session.sessionPreset = .high
let input = try AVCaptureDeviceInput(device: dev)
session.addInput(input)
let output = AVCaptureVideoDataOutput()
output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
let g = Grabber()
output.setSampleBufferDelegate(g, queue: DispatchQueue(label: "frames"))
session.addOutput(output)
session.startRunning()
var waited = 0.0                                            // wireless cameras (Continuity) need seconds to start
while waited < 20.0 {
    g.lock.lock(); let ok = g.latest != nil; g.lock.unlock()
    if ok { break }
    Thread.sleep(forTimeInterval: 0.25); waited += 0.25
}
print("first frame after", waited, "s")
if #available(macOS 14.0, *) { print("device suspended:", dev.isSuspended) }
print("active format:", dev.activeFormat.formatDescription)
g.lock.lock(); let probe = g.latest; g.lock.unlock()
if let pb = probe {
    CVPixelBufferLockBaseAddress(pb, .readOnly)
    let bpr = CVPixelBufferGetBytesPerRow(pb), h = CVPixelBufferGetHeight(pb)
    let base = CVPixelBufferGetBaseAddress(pb)!.assumingMemoryBound(to: UInt8.self)
    var sum = 0, mx = 0
    for i in stride(from: 0, to: bpr * h, by: 997) { sum += Int(base[i]); mx = max(mx, Int(base[i])) }
    CVPixelBufferUnlockBaseAddress(pb, .readOnly)
    print("raw buffer: format", CVPixelBufferGetPixelFormatType(pb), "bytes/row", bpr, "height", h,
          "sampled mean", Double(sum) / Double(bpr * h / 997), "max", mx)
}
Thread.sleep(forTimeInterval: 2.0)                          // let auto-exposure settle
let ctx = CIContext()
for i in 0..<count {
    g.lock.lock(); let pb = g.latest; g.lock.unlock()
    guard let buf = pb, let cg = ctx.createCGImage(CIImage(cvPixelBuffer: buf), from: CIImage(cvPixelBuffer: buf).extent) else {
        print("no frame yet"); Thread.sleep(forTimeInterval: interval); continue
    }
    let stamp = ISO8601DateFormatter().string(from: Date())
    let url = URL(fileURLWithPath: outDir).appendingPathComponent(String(format: "frame_%04d.png", i))
    let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, cg, nil); CGImageDestinationFinalize(dst)
    print("saved", url.lastPathComponent, cg.width, "x", cg.height, stamp)
    fflush(stdout)
    if i < count - 1 { Thread.sleep(forTimeInterval: interval) }
}
session.stopRunning()
