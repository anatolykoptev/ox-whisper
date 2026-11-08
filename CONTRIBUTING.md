# Contributing

Bug reports, fixes and focused features are welcome. For a larger change, open an issue first so
the approach can be agreed before you write it. Report a vulnerability privately, as described in
[SECURITY.md](SECURITY.md), not in an issue.

## Development

The service targets aarch64 Linux: the sherpa-onnx libraries under `vendor/` are prebuilt for
ARM64. Rust 1.88 (edition 2024) is what CI and the Dockerfile use, and
[cargo-nextest](https://nexte.st) runs the tests.

```sh
git clone https://github.com/anatolykoptev/ox-whisper.git
cd ox-whisper
export SHERPA_LIB_PATH=$PWD/vendor/sherpa-onnx
export LD_LIBRARY_PATH=$PWD/vendor/sherpa-onnx/lib
cargo fmt --check
cargo nextest run --workspace --locked
docker build .          # the full image, as the CI build job does
```

These are the three jobs in `.github/workflows/ci.yml` (`fmt`, `test`, `build`); if they pass
locally they pass there. `make check` runs `cargo fmt`, clippy and the tests, and clippy is not part
of the gate. A pull request that changes only documents skips the CI jobs. Running the server needs
the models: `scripts/download-models.sh` fetches them (see the README).

## Commits

[Conventional Commits](https://www.conventionalcommits.org): `feat:`, `fix:`, `perf:`, `docs:`,
`chore:`, `ci:`. release-please reads them on `main`, opens a release PR that bumps the version and
`CHANGELOG.md`, and tags on merge. Do not edit the version, `CHANGELOG.md` or
`.release-please-manifest.json` by hand, and do not tag manually. Put a breaking change in a
`BREAKING CHANGE:` footer.

## Pull requests

One concern per pull request. State what changed and how you tested it, and add a test for every
behaviour change; a bug fix needs a test that fails without it. Do not commit models, audio you do
not have the right to share, hostnames, credentials or personal data. Add new documents to
[docs/INDEX.md](docs/INDEX.md).
