#
# PhoenixDB — iOS podspec.
#
# The Rust static library is compiled from source during `pod install` /
# `flutter build`, because Apple platforms require code signing and an Xcode
# SDK that only exist on the developer's Mac. `script_phase` below invokes
# rust/build-apple.sh, which produces a static archive for the active
# architecture and links it into the app.
#
# Requirements on the build machine:
#   rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios
#
Pod::Spec.new do |s|
  s.name             = 'phoenixdb'
  s.version          = '0.1.0'
  s.summary          = 'ACID-compliant embedded key/value engine (Rust + dart:ffi).'
  s.description      = <<-DESC
B+Tree index, MVCC snapshot isolation, write-ahead log and CRC32-checksummed
pages, implemented in Rust and exposed to Dart through a zero-overhead FFI
layer.
                       DESC
  s.homepage         = 'https://github.com/phoenixdb/phoenixdb'
  s.license          = { :type => 'BSD-3-Clause', :file => '../LICENSE' }
  s.author           = { 'PhoenixDB Authors' => 'phoenixdb@example.com' }
  s.source           = { :path => '.' }

  s.dependency 'Flutter'
  s.platform = :ios, '12.0'

  # A placeholder source file is required: CocoaPods will not vendor a pod
  # that compiles nothing, and the real code arrives as a static archive.
  s.source_files = 'Classes/**/*'

  # Force the linker to keep every exported symbol. Without this the static
  # archive is dead-stripped, because nothing in the Objective-C world
  # references phoenix_* directly - only dlsym from Dart does.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'OTHER_LDFLAGS' => '-force_load ${PODS_TARGET_SRCROOT}/libphoenixdb.a',
    'STRIP_STYLE' => 'non-global',
  }
  s.user_target_xcconfig = {
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
  }

  s.script_phase = {
    :name => 'Build PhoenixDB Rust library',
    :script => '"${PODS_TARGET_SRCROOT}/../rust/build-apple.sh" ios',
    :execution_position => :before_compile,
    :output_files => ['${PODS_TARGET_SRCROOT}/libphoenixdb.a'],
  }
end
