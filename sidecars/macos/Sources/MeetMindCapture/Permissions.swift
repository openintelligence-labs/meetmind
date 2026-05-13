// Permissions.swift — TCC probes for mic + screen recording.
// Returns one of "granted" / "denied" / "not_determined" / "restricted".

import AVFoundation
import CoreGraphics
import Foundation

enum PermissionState: String {
    case granted        = "granted"
    case denied         = "denied"
    case notDetermined  = "not_determined"
    case restricted     = "restricted"
}

enum Permissions {
    static func microphone() -> PermissionState {
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:   return .granted
        case .denied:       return .denied
        case .notDetermined: return .notDetermined
        case .restricted:   return .restricted
        @unknown default:   return .notDetermined
        }
    }

    /// Screen recording permission gates ScreenCaptureKit and (since macOS 15)
    /// system-wide audio taps. We probe via CGRequestScreenCaptureAccess.
    static func screenRecording() -> PermissionState {
        if CGPreflightScreenCaptureAccess() {
            return .granted
        }
        return .denied
    }
}
