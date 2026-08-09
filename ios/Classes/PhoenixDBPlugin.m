// Placeholder translation unit.
//
// CocoaPods requires a pod to contain at least one compilable source file.
// All real functionality lives in the Rust static archive (libphoenixdb.a),
// which is produced by rust/build-apple.sh and force-loaded by the linker
// settings in phoenixdb.podspec.
//
// The symbol below exists only so the object file is non-empty; it is never
// called from Dart.

#import <Foundation/Foundation.h>

__attribute__((visibility("default")))
const char *phoenixdb_ios_placeholder(void) {
    return "phoenixdb";
}
