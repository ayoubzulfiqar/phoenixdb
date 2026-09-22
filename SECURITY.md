# Security Policy

## Supported versions

Security fixes are released for the latest minor version. Upgrade to receive
them; older lines are not patched.

| Version | Supported |
| --- | --- |
| 4.0.x | Yes |
| < 4.0 | No |

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Report it privately through GitHub: open the repository's
[Security tab](https://github.com/ayoubzulfiqar/phoenixdb/security) and choose
**Report a vulnerability**. Only the maintainers can see the report.

A useful report includes:

- the PhoenixDB version and platform (OS, CPU architecture, Dart/Flutter
  version);
- what an attacker controls — database files, keys and values, SQL text,
  FFI arguments, documents fed to the AI toolkit — and what they gain;
- a minimal reproduction, ideally a failing test or a crafted input file.

The maintainers aim to acknowledge a report within a week, agree on a fix and a
disclosure date with the reporter, and credit the reporter in the release
notes unless they prefer otherwise.

## Scope

In scope:

- **Memory safety at the C ABI** (`rust/src/ffi/`): every entry point must
  validate pointers, lengths and handles before use and must never unwind into
  the caller. A crash, out-of-bounds access or use-after-free reachable through
  the public ABI or the Dart bindings is a vulnerability.
- **Hostile database files:** opening a corrupted or crafted file must fail
  with an error, never read or write out of bounds, hang, or exhaust memory.
- **SQL parameter binding:** a bound parameter (`?`) must never be interpreted
  as SQL.
- **Durability and isolation guarantees** that are documented but can be
  broken — for example committed data lost after a crash, or one transaction
  observing another's uncommitted writes.
- **The AI toolkit** (`lib/src/ai/`): API keys must only ever be sent to the
  configured endpoint.

Not vulnerabilities in PhoenixDB:

- Prompt injection through documents you retrieve and pass to a model. Treat
  retrieved text as untrusted input to the model, as with any RAG system.
- The standalone `security::encryption`, `security::rbac` and
  `security::audit` modules are libraries that the engine does not yet route
  through: a database is **not** encrypted at rest and access is **not**
  role-checked today. Bugs inside those modules are still welcome as reports.
- Denial of service by a caller that already has full access to the database
  file or the process.
