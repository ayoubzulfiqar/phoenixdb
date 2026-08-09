#
# PhoenixDB — macOS podspec.
#
# Like the iOS pod, the Rust static archive is compiled during the Xcode build
# via rust/build-apple.sh, producing a universal (arm64 + x86_64) binary.
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

  s.dependency 'FlutterMacOS'
  s.platform = :osx, '10.14'

  s.source_files = 'Classes/**/*'

  # -force_load keeps the phoenix_* symbols alive; nothing in the Objective-C
  # side references them, so the linker would otherwise strip the archive.
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'OTHER_LDFLAGS' => '-force_load ${PODS_TARGET_SRCROOT}/libphoenixdb.a',
    'STRIP_STYLE' => 'non-global',
  }

  s.script_phase = {
    :name => 'Build PhoenixDB Rust library',
    :script => '"${PODS_TARGET_SRCROOT}/../rust/build-apple.sh" macos',
    :execution_position => :before_compile,
    :output_files => ['${PODS_TARGET_SRCROOT}/libphoenixdb.a'],
  }
end
