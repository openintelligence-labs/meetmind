// swift-tools-version:5.9
//
// MeetMind macOS native sidecars.
//
// Two executables share this package:
//   • meetmind-capture-macos — bot-free system + mic audio capture
//     (Core Audio Tap + AVAudioEngine; protocol in SPEC_CAPTURE_IPC.md)
//   • meetmind-stt-macos     — Parakeet TDT 0.6B v3 via FluidAudio
//     (CoreML/ANE; protocol in SPEC_STT_IPC.md)
//
// Both speak the same length-prefixed binary frame format defined in
// `meetmind.ipc.protocol`.
//
// Build:
//   swift build -c release
// Outputs:
//   .build/release/meetmind-capture-macos
//   .build/release/meetmind-stt-macos

import PackageDescription

let package = Package(
    name: "MeetMindSidecars",
    platforms: [
        .macOS(.v14),  // Core Audio Tap requires 14.4+; FluidAudio requires 14+
    ],
    products: [
        .executable(name: "meetmind-capture-macos", targets: ["MeetMindCapture"]),
        .executable(name: "meetmind-stt-macos", targets: ["MeetMindSTT"]),
        .executable(name: "meetmind-diar-macos", targets: ["MeetMindDiar"]),
    ],
    dependencies: [
        // FluidAudio: Parakeet TDT v3 + Whisper + diarization on Apple Silicon.
        // Pinned to the v0.14.x line which is what we benchmarked against
        // in the architecture research.
        .package(url: "https://github.com/FluidInference/FluidAudio.git", from: "0.14.0"),
    ],
    targets: [
        .target(
            name: "MeetMindIPC",
            path: "Sources/MeetMindIPC"
        ),
        .executableTarget(
            name: "MeetMindCapture",
            dependencies: ["MeetMindIPC"],
            path: "Sources/MeetMindCapture"
        ),
        .executableTarget(
            name: "MeetMindSTT",
            dependencies: [
                "MeetMindIPC",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/MeetMindSTT"
        ),
        .executableTarget(
            name: "MeetMindDiar",
            dependencies: [
                "MeetMindIPC",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Sources/MeetMindDiar"
        ),
    ]
)
