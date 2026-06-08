# Public launch checklist

Use this checklist when validating a clean public Fathom checkout or preparing a launch-facing handoff. It is scoped to the current local, no-download public contract plus optional networked backend acceptance evidence.

## 1. Clean clone and install

```bash
git clone https://github.com/timtoole02/Fathom/ fathom
cd fathom
npm --prefix frontend ci
cargo test -q
```

This confirms the ordinary Rust and frontend toolchains are present, with frontend dependencies resolved from the checked-in lockfile. It does not download catalog model fixtures or enable the non-default ONNX Runtime feature.

## 2. No-download verification gates

Run these before treating docs/API contract changes as launch-ready:

```bash
git diff --check
python3 -m py_compile \
  examples/api/openai-sdk.py \
  examples/api/python-no-deps.py \
  scripts/api_client_examples_regression.py \
  scripts/backend_acceptance_artifact_qa.py \
  scripts/bench_backend.py \
  scripts/ci_static_policy.py \
  scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py \
  scripts/public_api_contract_qa.py \
  scripts/public_contract_smoke_artifact_qa.py \
  scripts/qwen25_optional_api_acceptance_artifact_qa.py \
  scripts/reference_llama3_tokenizer_ids.py \
  scripts/smollm2_optional_api_acceptance_artifact_qa.py
python3 scripts/ci_static_policy.py
python3 scripts/ci_static_policy.py --self-test
python3 scripts/api_client_examples_regression.py
python3 scripts/api_client_examples_regression.py --self-test
python3 scripts/public_api_contract_qa.py
python3 scripts/public_api_contract_qa.py --self-test
python3 scripts/public_contract_smoke_artifact_qa.py
python3 scripts/backend_acceptance_artifact_qa.py
python3 scripts/minilm_embeddings_optional_api_acceptance_artifact_qa.py
python3 scripts/smollm2_optional_api_acceptance_artifact_qa.py
python3 scripts/qwen25_optional_api_acceptance_artifact_qa.py
bash -n examples/api/curl-quickstart.sh
bash -n scripts/public_risk_scan.sh
bash -n scripts/public_api_contract_smoke.sh
bash -n scripts/backend_acceptance_smoke.sh
bash -n scripts/minilm_embeddings_optional_api_acceptance_smoke.sh
bash -n scripts/smollm2_optional_api_acceptance_smoke.sh
bash -n scripts/qwen25_optional_api_acceptance_smoke.sh
bash -n scripts/smoke.sh
bash -n scripts/start-backend.sh
bash -n scripts/start.sh
bash -n scripts/stop.sh
bash scripts/public_risk_scan.sh --self-test
bash scripts/public_risk_scan.sh
npm --prefix frontend run build
npm --prefix frontend run qa:copy
```

For the real backend no-download contract check, run:

```bash
bash scripts/public_api_contract_smoke.sh
```

That smoke starts `fathom-server` with isolated temporary state/model/log directories and checks the public `/v1` routing/refusal boundary from [`docs/api/public-contract.json`](api/public-contract.json) and [`docs/api/v1-contract.md`](api/v1-contract.md), including JSON refusals for unsupported `/v1` routes and methods. It does not install catalog models, download fixtures, enable ONNX features, call providers, or prove model quality.

To keep a share-safe pass/fail summary for release handoff, set `FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR` to a directory you control:

```bash
FATHOM_PUBLIC_CONTRACT_ARTIFACT_DIR=public-contract-artifacts \
  bash scripts/public_api_contract_smoke.sh
```

The optional `public-contract-smoke-summary.json` and `.md` files include commit, manifest path/name/status, endpoint checks, boundary checks, and scope caveats only. They intentionally omit local temp paths, server log tails, request secrets, and model/provider payloads. Validate them offline before sharing with `python3 scripts/public_contract_smoke_artifact_qa.py public-contract-artifacts`.

The static QA also requires root `.gitattributes` text-normalization metadata so public diffs stay stable across platforms while binary model, package, mobile, media, and local plan artifact extensions remain marked as binary.

## 3. Backend/API quick smoke

For a backend-only manual pass:

```bash
bash scripts/start-backend.sh
BASE=http://127.0.0.1:8180
curl -fsS "$BASE/v1/health" | python3 -m json.tool
curl -fsS "$BASE/v1/models" | python3 -m json.tool
bash scripts/stop.sh
```

Use [`docs/api/backend-only-quickstart.md`](api/backend-only-quickstart.md) for catalog install examples, TinyStories chat, MiniLM embeddings, retrieval, and expected refusals. Those catalog demos require network access for pinned fixture downloads. The backend API has no built-in authentication and is intended for loopback development; do not expose it directly to the internet or an untrusted LAN without your own access controls and a [`SECURITY.md`](../SECURITY.md) review.
For a compact refusal/non-claim checklist before Phase 16 work, see [`docs/api/refusal-boundary-matrix.md`](api/refusal-boundary-matrix.md).

## 4. Optional networked acceptance smoke

When you need fuller backend evidence and network access is available:

```bash
FATHOM_ACCEPTANCE_KEEP_ARTIFACTS=1 bash scripts/backend_acceptance_smoke.sh
```

Review `summary.md` first, then `summary.json` and the named JSON artifacts. Keep `summary.local.json` private unless you have reviewed it; local paths and runner-specific details belong there. Before sharing artifacts publicly, run `bash scripts/public_risk_scan.sh` and manually inspect logs/full payloads for local paths or request text. The scanner checks tracked-file privacy patterns including macOS, Linux, and Windows home/profile paths, secret-token/private-key/cloud API-key patterns, dependency lockfiles for local file/path, SSH-only, authenticated URL, secret, and private local source entries, tracked Git submodule metadata for local/relative, SSH-only, authenticated, secret, or private source URLs, public overclaim patterns, oversized tracked files, tracked Git LFS pointer files that can hide external artifact downloads, tracked OS/platform metadata files such as `.DS_Store`, `Thumbs.db`, `ehthumbs.db`, `desktop.ini`, `$RECYCLE.BIN/`, `__MACOSX/`, `._*`, `.AppleDouble`, `.fseventsd/`, `.Spotlight-V100/`, `.TemporaryItems/`, `.Trashes/`, `.LSOverride`, and `.localized`, root `.gitignore` coverage for local OS/platform metadata files, tracked editor backup/swap files, local editor history directories, and patch/diff files, root `.gitignore` coverage for local editor backup/swap files plus local editor history directories and patch/diff files, tracked IDE workspace/config artifacts such as `.zed/`, root and nested `.gitignore` coverage for local IDE workspace/config artifacts, tracked credential/config filenames including SSH private-key filenames, `.ssh/` directories, direnv config/state, Git credential/config files, generic secret material paths, TLS certificate/request artifacts, and Java/Android/Apple signing key material, root and nested `.gitignore` coverage for local credential/config files including SSH private-key filenames, `.ssh/`, direnv config/state, Git credential/config files, generic secret material patterns, TLS certificate/request artifact patterns, and Java/Android/Apple signing key material patterns, tracked local cloud SDK credential/config files, root and nested `.gitignore` coverage for local cloud SDK credential/config files, tracked Kubernetes credential/config files, root and nested `.gitignore` coverage for local Kubernetes credential/config files, tracked workspace/personal agent context files including local assistant state such as `.codex/`, `.claude/`, `.continue/`, `.cursor/`, `.windsurf/`, and `.aider.*`, root and nested `.gitignore` coverage for personal workspace context including local assistant state, tracked local shell/REPL command history files, root and nested `.gitignore` coverage for local shell/REPL command history files, tracked local runtime/artifact detail files including PID/process marker files, DuckDB databases/WAL files, and Redis RDB snapshots, root and nested `.gitignore` coverage for local runtime/artifact detail files, tracked local log/trace/profiling/debug-output artifacts including SARIF/code-scanning reports and crash dumps, root `.gitignore` coverage for local log/trace/profiling/debug-output artifacts including SARIF/code-scanning reports and crash dumps, tracked Python cache/build artifacts including Hypothesis example databases, root and nested `.gitignore` coverage for local Python cache/build artifacts, tracked Python virtualenv/dependency artifacts including local uv/pip package-manager caches, root and nested `.gitignore` coverage for local Python virtualenv/dependency artifacts including local uv/pip package-manager caches, tracked frontend/Node cache/build artifacts including root-level `dist/` or `build/` output, package-manager caches/stores such as `.bun/`, `.npm/`, `.pnpm-store/`, selected `.yarn/` cache/state paths, Bun debug logs, Vite/Vitest cache/config timestamp artifacts, Storybook static build output, and lint cache files such as `.eslintcache` and `.stylelintcache`, root and nested `.gitignore` coverage for local frontend/Node cache/build artifacts, tracked local cache artifacts such as `.cache/`, root `.gitignore` coverage for local cache artifacts, tracked local temporary/scratch artifacts, root and nested `.gitignore` coverage for local temporary/scratch artifacts, tracked local test report/coverage artifacts including Playwright/browser-test report directories, Python coverage data files, NYC/Istanbul coverage artifacts, and LCOV outputs, root and nested `.gitignore` coverage for local test report/coverage artifacts including Playwright/browser-test report directories, Python coverage data files, NYC/Istanbul coverage artifacts, and LCOV outputs, tracked local notebook checkpoint/runtime config artifacts, tracked notebook execution outputs, root and nested `.gitignore` coverage for local notebook artifacts, tracked Rust/Cargo cache/build artifacts, root and nested `.gitignore` coverage for local Rust/Cargo cache/build artifacts, tracked local Elixir/Mix build/dependency artifacts, root `.gitignore` coverage for local Elixir/Mix build/dependency artifacts, tracked local native/CMake build artifacts, root and nested `.gitignore` coverage for local native/CMake build artifacts, tracked release/package artifacts including common archive and installer formats, root `.gitignore` coverage for local release/package artifacts, tracked backup/dump artifacts, root `.gitignore` coverage for local backup/dump artifacts, tracked local model/checkpoint artifacts, root `.gitignore` coverage for local model/checkpoint artifacts, tracked local ML experiment/tracking artifacts, root and nested `.gitignore` coverage for local ML experiment/tracking artifacts, tracked local Docker/container artifacts, root `.gitignore` coverage for local Docker/container artifacts, tracked local deployment platform artifacts such as `.vercel/` and `.netlify/`, root and nested `.gitignore` coverage for local deployment platform artifacts, tracked local Terraform/OpenTofu/Terragrunt state artifacts, root and nested `.gitignore` coverage for local infrastructure state artifacts, tracked local Nix build result artifacts, root `.gitignore` coverage for local Nix build result artifacts, tracked local Bazel build artifacts, root `.gitignore` coverage for local Bazel build artifacts, tracked local mobile/Xcode/Android build artifacts, root `.gitignore` coverage for local mobile/Xcode/Android build artifacts, tracked local mobile/Xcode/Android signing/provisioning artifacts, root `.gitignore` coverage for local mobile/Xcode/Android signing/provisioning artifacts, tracked local screenshot/screen-recording artifacts, root `.gitignore` coverage for local screenshot/screen-recording artifacts, tracked local audio/video capture/export artifacts, root `.gitignore` coverage for local audio/video capture/export artifacts, and tracked symlinks that escape the repository or resolve only to missing/untracked local targets; it is not a complete privacy audit.

It also blocks tracked Deno local cache artifacts such as `.deno/` and `deno-dir/`, with matching root and nested `.gitignore` coverage for local Deno cache artifacts. It does not treat source-of-truth Deno project files such as `deno.json`, `deno.jsonc`, `deno.lock`, JavaScript files, and TypeScript files as local tool artifacts.

It also blocks tracked local frontend/static-site framework caches and build outputs such as Astro `.astro/`, Docusaurus `.docusaurus/`, Hugo `resources/_gen/` and `.hugo_build.lock`, Jekyll `_site/`, `.jekyll-cache/`, and `.sass-cache/`, Vite `.vite/`, VitePress `.vitepress/cache/` and `.vitepress/dist/`, Metro `.metro-cache/`, Nx `.nx/`, SWC `.swc/`, Rollup TypeScript plugin `.rpt2_cache/`, Webpack `.webpack-cache/`, Next.js `.next/`, SvelteKit `.svelte-kit/`, Nuxt `.nuxt/` and `.output/`, and Angular `.angular/cache/` directories, with matching root and nested `.gitignore` coverage for those local frontend/static-site artifacts. It also rejects selected nested Yarn cache/state artifacts such as `.yarn/cache/`, `.yarn/unplugged/`, `.yarn/build-state.yml`, and `.yarn/install-state.gz` while preserving source-of-truth lockfiles and package manifests. It does not treat source-of-truth Metro configuration such as `metro.config.js` as a local cache artifact.

It also blocks tracked local Lighthouse/LHCI audit outputs such as `.lighthouseci/`, `lhci_reports/`, `lighthouse-report.html`, and `lighthouse-report.json`, with matching root and nested `.gitignore` coverage for generated audit outputs while preserving source-of-truth Lighthouse config files such as `.lighthouserc.*`.

It also blocks tracked Watchman local state cookies such as `.watchman-cookie` and `.watchman-cookie-*`, with matching root and nested `.gitignore` coverage through the local cache artifact guard. It does not treat source-of-truth Watchman configuration such as `.watchmanconfig` as a local cache artifact.

It also blocks tracked local `.cache/` directories at any tree depth, with matching root and nested `.gitignore` coverage through the local cache artifact guard.

It also blocks tracked local documentation build artifacts such as LaTeX auxiliary files (`*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.synctex.gz`) and `_minted-*` caches, with matching root and nested `.gitignore` coverage for local documentation build artifacts. It intentionally does not block all PDFs because some PDFs can be source-of-truth public docs rather than rebuildable local output.

It also blocks tracked local JVM dependency artifacts such as `.m2/` at any tree depth, with matching root and nested `.gitignore` coverage for local JVM dependency artifacts. It does not treat source-of-truth Maven/JVM project files such as `pom.xml`, Java source files, Kotlin source files, Scala source files, or JVM docs as local dependency artifacts.

It also blocks tracked local Clojure/Leiningen artifacts such as `.lein/`, `.cpcache/`, and `.shadow-cljs/` at any tree depth, plus `.nrepl-port`, with matching root and nested `.gitignore` coverage for local Clojure/Leiningen artifacts. It does not treat source-of-truth Clojure files such as `project.clj`, `deps.edn`, `bb.edn`, `shadow-cljs.edn`, or Clojure/ClojureScript source files as local tool artifacts.

It also blocks tracked local CI runner artifacts/config such as `.act/`, `.actrc`, `actions-runner/`, `_work/`, `_diag/`, `.runner`, `.credentials`, and `.credentials_rsaparams`, with matching root and nested `.gitignore` coverage, so local workflow runner state cannot be mistaken for launch evidence.

It also blocks tracked local Swift Package Manager build/workspace artifacts such as `.build/` and `.swiftpm/` at any tree depth, with matching root and nested `.gitignore` coverage for local Swift Package Manager artifacts. It does not treat source-of-truth Swift package files such as `Package.swift` or Swift source files as local tool artifacts.

It also blocks tracked local Zig build artifacts such as `.zig-cache/`, `zig-cache/`, and `zig-out/` at any tree depth, with matching root and nested `.gitignore` coverage for local Zig build artifacts. It does not treat source-of-truth Zig files such as `build.zig`, `build.zig.zon`, or Zig source files as local tool artifacts.

It also blocks tracked local Dart/Flutter artifacts such as `.dart_tool/`, `.pub-cache/`, `.pub/`, `.packages`, `.flutter-plugins`, and `.flutter-plugins-dependencies` at any tree depth, with matching root and nested `.gitignore` coverage for local Dart/Flutter artifacts. It does not treat source-of-truth Dart/Flutter files such as `pubspec.yaml`, `pubspec.lock`, Dart source, Android Gradle project files, or iOS Xcode project files as local tool artifacts.

The native/CMake build artifact guard rejects tracked local native/CMake build/user-local artifacts such as `cmake-build-*`, `CMakeFiles/` at any tree depth, CTest `Testing/Temporary/` output, CPack staging output such as `_CPack_Packages/`, generated CTest/Dart files such as `CTestTestfile.cmake` and `DartConfiguration.tcl`, generated CPack files such as `CPackConfig.cmake` and `CPackSourceConfig.cmake`, `CMakeCache.txt`, user-local `CMakeUserPresets.json`, `cmake_install.cmake`, `compile_commands.json`, `install_manifest.txt`, and Ninja local state files such as `.ninja_deps` and `.ninja_log`, with matching root and nested `.gitignore` coverage for local native/CMake build artifacts. It does not treat source-of-truth CMake files such as `CMakeLists.txt` and `CMakePresets.json` as local tool artifacts.

It also blocks tracked Expo local project state such as `.expo/` and `.expo-shared/` at any tree depth, with matching root and nested `.gitignore` coverage through the local mobile/Xcode/Android build artifact guard. It does not treat source-of-truth mobile app code, Expo config, Android Gradle project files, or iOS Xcode project files as local tool artifacts.

It also blocks tracked Android native build intermediates such as `.cxx/` and `.externalNativeBuild/` at any tree depth, including the same directories under `android/`, with matching root, nested, and `android/` `.gitignore` coverage through the local mobile/Xcode/Android build artifact guard. It does not treat source-of-truth Android Gradle project files, JNI/C++ source files, or checked-in native build configuration as local tool artifacts.

It also blocks tracked CocoaPods dependency outputs such as root `Pods/` and `ios/Pods/`, with matching root `.gitignore` coverage through the local mobile/Xcode/Android build artifact guard. It does not treat source-of-truth CocoaPods files such as `Podfile` or `Podfile.lock` as local tool artifacts.

It also blocks tracked Carthage dependency outputs such as `Carthage/Build/` and `Carthage/Checkouts/`, with matching root `.gitignore` coverage through the local mobile/Xcode/Android build artifact guard. It does not treat source-of-truth Carthage files such as `Cartfile` or `Cartfile.resolved` as local tool artifacts.

It also blocks tracked Fastlane generated report/test artifacts such as `fastlane/report.xml`, `fastlane/Preview.html`, and `fastlane/test_output/`, with matching root `.gitignore` coverage through the local mobile/Xcode/Android build artifact guard. It does not treat source-of-truth Fastlane configuration or App Store metadata such as `Fastfile`, `Appfile`, or `fastlane/metadata/` as local tool artifacts.

The general mobile/Xcode/Android build artifact guard blocks tracked local build outputs such as `DerivedData/`, `.gradle/`, `xcuserdata/`, `local.properties`, `.xcuserstate`, `.xcresult`, `.ipa`, `.apk`, `.aab`, and `.dSYM` files, and requires root, nested, Android-specific, and platform artifact `.gitignore` coverage for local mobile build outputs.

The mobile/Xcode/Android signing/provisioning artifact guard also blocks tracked local provisioning/build archives such as `*.mobileprovision`, `*.provisionprofile`, and `*.xcarchive`, with matching root `.gitignore` coverage for local mobile/Xcode/Android signing/provisioning artifacts.

The screenshot/screen-recording artifact guard blocks tracked local captures with default macOS capture prefixes such as `Screenshot *`, `Screen Shot *`, and `Screen Recording *`, with matching root `.gitignore` coverage for local screenshot/screen-recording artifacts.

The audio/video capture/export artifact guard blocks tracked local captures and exports with common media extensions and default voice memo/audio recording prefixes such as `Audio Recording *`, `Voice Memo *`, `*.m4a`, `*.mov`, `*.mp3`, `*.mp4`, and `*.wav`, with matching root `.gitignore` coverage for local audio/video capture/export artifacts.

It also blocks tracked local Bazel output symlinks/directories such as `bazel-bin/`, `bazel-out/`, `bazel-testlogs/`, and root `bazel-*` outputs, with matching root `.gitignore` coverage for local Bazel build artifacts. It does not treat source-of-truth Bazel files such as `BUILD`, `BUILD.bazel`, `MODULE.bazel`, or `.bzl` files as build artifacts.

It also blocks tracked local Buck/Buck2 build artifacts such as `.buckd/` and `buck-out/`, with matching root and nested `.gitignore` coverage for local Buck/Buck2 build artifacts. It does not treat source-of-truth Buck files such as `BUCK`, `BUCK.v2`, or `.buckconfig` as local build artifacts.

For Yarn lockfiles, local `portal:` dependencies and local `patch:` references count as local/private dependency sources.

The Rust/Cargo guard also rejects tracked Rust/Cargo cache/build artifacts such as `.cargo/`, `target/`, compiler outputs, and coverage profiles, with matching root and nested `.gitignore` coverage.

The Meson build artifact guard also rejects tracked local Meson build artifacts such as `.mesonpy-*`, `meson-info/`, `meson-logs/`, and `meson-private/`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Meson files such as `meson.build`, `meson_options.txt`, or `meson.options` as local tool artifacts.

The Autotools build artifact guard also rejects tracked local Autotools configure and Libtool outputs such as `autom4te.cache/`, `.deps/`, `.libs/`, `config.log`, `config.status`, generated `libtool` files, `*.lo`, and `*.la`, with matching root `.gitignore` coverage. It does not treat source-of-truth Autotools files such as `configure.ac`, `Makefile.am`, or Autotools docs as local build artifacts.

The SCons build artifact guard also rejects tracked local SCons configuration/cache outputs such as `.sconf_temp/`, `sconf_temp/`, `.sconsign`, and `.sconsign.dblite`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth `SConstruct` or `SConscript` files as local build artifacts.

The release/package artifact guard also rejects tracked local release/package artifacts including common archive formats such as `.zip`, `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, `.tar.xz`, `.7z`, `.rar`, and `.zst`; installer/package formats such as `.dmg`, `.pkg`, `.whl`, `.egg`, `.deb`, `.rpm`, and `.msi`; JVM package archives such as `.jar`, `.war`, and `.ear`; and root `artifacts/` outputs, with matching root `.gitignore` coverage.

The backup/dump artifact guard also rejects tracked local backup and data dump artifacts such as root `backups/`, root `dumps/`, `*.bak`, `*.backup`, `*.dump`, and `*.sql`, with matching root `.gitignore` coverage.

The model/checkpoint artifact guard also rejects tracked local model/checkpoint artifacts such as top-level `models/`, `model-store/`, `weights/`, `checkpoints/`, and model binary/data files including `.safetensors`, `.gguf`, `.onnx`, `.bin`, `.pt`, `.pth`, `.ckpt`, `.npz`, `.npy`, and `.tflite`, with matching root `.gitignore` coverage.

The JVM dependency artifact guard also rejects tracked local Maven repository/cache state such as `.m2/` at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Maven/JVM project files such as `pom.xml`, Java source files, Kotlin source files, Scala source files, or JVM docs as local dependency artifacts.

The JVM compiler artifact guard also rejects tracked local bytecode outputs such as `*.class` and `*.tasty`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth JVM project files such as `pom.xml`, `build.gradle`, `build.sbt`, Java source files, Kotlin source files, or Scala source files as local tool artifacts.

The .NET/NuGet artifact guard also rejects tracked local dependency/build/user-state artifacts such as `.nuget/`, `packages/`, `bin/Debug/`, `obj/Release/`, `project.assets.json`, `project.nuget.cache`, `*.nupkg`, `*.snupkg`, `*.csproj.user`, and `*.suo`, with matching root `.gitignore` coverage. It does not treat source-of-truth .NET files such as `*.csproj`, `*.fsproj`, `*.vbproj`, `*.sln`, `Directory.Build.props`, C# source files, F# source files, or NuGet lock files as local tool artifacts.

The Gradle/JVM build artifact guard also rejects tracked local Gradle cache/state such as `.gradle/` at any tree depth and common Gradle `build/` output subtrees such as `classes/`, `reports/`, `test-results/`, `tmp/`, `generated/`, `intermediates/`, and `libs/`, with matching root and nested `.gitignore` coverage for Gradle cache/state. It does not treat source-of-truth Gradle project files such as `build.gradle`, `settings.gradle`, `gradle.properties`, `gradlew`, or `gradle/wrapper/gradle-wrapper.properties` as build artifacts.

The Kotlin/Kotlin Native compiler artifact guard also rejects tracked local Kotlin/Kotlin Native compiler artifacts such as `.kotlin/` and `.konan/` at any tree depth, with matching root and nested `.gitignore` coverage for local Kotlin/Kotlin Native artifacts. It does not treat source-of-truth Kotlin files such as `build.gradle.kts`, `settings.gradle.kts`, Kotlin source files, or Gradle version catalog files as local tool artifacts.

The Scala/SBT build artifact guard also rejects tracked local Scala build server/IDE outputs such as `.bloop/`, `.bsp/`, `.metals/`, and `.scala-build/` at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Scala/SBT files such as `build.sbt`, `project/build.properties`, or Scala source files as local tool artifacts.

The Haskell Stack/Cabal build artifact guard also rejects tracked local build outputs such as `.stack-work/`, `dist-newstyle/`, `.cabal-sandbox/`, and `cabal.sandbox.config` at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Haskell files such as `.cabal` files, `stack.yaml`, `cabal.project`, or Haskell source files as local Stack/Cabal build artifacts by filename alone.

The OCaml/opam local switch artifact guard also rejects tracked local switch/dependency artifacts such as `_opam/` and `.opam-switch/` at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth OCaml project files such as `dune`, `dune-project`, `*.opam`, or OCaml source files as local tool artifacts.

The Lua/LuaRocks artifact guard also rejects tracked local dependency/build/package artifacts such as `.luarocks/`, `lua_modules/`, and `*.rock` at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Lua files such as `*.lua`, `*.rockspec`, LuaRocks lock files, or Lua config files as local tool artifacts.

The Ruby/Bundler guard also rejects tracked local dependency artifacts such as `.bundle/`, `vendor/bundle/`, and `vendor/cache/` at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Ruby files such as `Gemfile`, `Gemfile.lock`, Ruby source files, Ruby docs, or non-artifact vendor paths as local Bundler dependency artifacts.

The PHP Composer guard also rejects tracked local dependency/test artifacts such as `vendor/autoload.php`, `vendor/bin/`, `vendor/composer/`, and `.phpunit.cache/` at any tree depth, plus `.phpunit.result.cache` and `composer.phar`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Composer files such as `composer.json`, `composer.lock`, PHP source files, or PHP docs as local dependency/test artifacts.

The Perl/CPAN build/dependency artifact guard rejects tracked local CPAN client, build, and local::lib outputs such as `.cpan/`, `.cpanm/`, `blib/`, `local/lib/perl5/`, `MYMETA.json`, `MYMETA.yml`, and `pm_to_blib`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Perl files such as `cpanfile`, `cpanfile.snapshot`, `Makefile.PL`, `Build.PL`, Perl source files, or Perl docs as local CPAN build/dependency artifacts.

The R/RStudio artifact guard also rejects tracked local session and dependency artifacts such as `.Rproj.user/` and `renv/library/` at any tree depth, plus `.Rhistory`, `.RData`, and `.Ruserdata`, with matching root and nested `.gitignore` coverage for local R/RStudio artifacts. It does not treat source-of-truth R files such as `*.R`, `*.Rmd`, `.Rproj` project files, `renv.lock`, `DESCRIPTION`, or R docs as local session/dependency artifacts.

The Julia depot/preference artifact guard also rejects tracked local Julia depot, preference, coverage, and allocation artifacts such as `.julia/` at any tree depth, `LocalPreferences.toml`, `*.jl.cov`, and `*.jl.mem`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Julia project files such as `Project.toml`, `Manifest.toml`, `Artifacts.toml`, or Julia source files as local depot/preference artifacts by filename alone.

The Go cache/test artifact guard also rejects tracked local generated outputs such as `.gocache/` and `.gomodcache/` at any tree depth, `cover.out`, `coverage.out`, `*.coverprofile`, and `*.test`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Go files such as `go.mod`, `go.sum`, Go source files, or Go docs as local cache/test artifacts.

The Elixir/Mix build/dependency artifact guard also rejects tracked local generated outputs such as `.elixir_ls/`, `_build/`, and `deps/` at any tree depth, with matching root and nested `.gitignore` coverage for local Elixir/Mix build/dependency artifacts. It does not treat source-of-truth Elixir/Mix files such as `mix.exs`, `mix.lock`, Elixir source files, or Elixir docs as local tool artifacts.

The Erlang/Rebar3 artifact guard also rejects tracked local cache and crash artifacts such as `.rebar3/` at any tree depth, `rebar3.crashdump`, and `erl_crash.dump`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Erlang/Rebar3 files such as `rebar.config`, `rebar.lock`, Erlang source files, or Erlang header files as local tool artifacts.

The Python cache/build artifact guard also rejects tracked local Pyre/Pytype and mypy daemon state such as `.pyre/`, `.pytype/`, and `.dmypy.json`, plus Python package metadata/build artifacts such as `.eggs/`, `*.egg-info/`, and `*.dist-info/`, with matching root and nested `.gitignore` coverage. The Python virtualenv/dependency artifact guard rejects local PDM/PEP 582 artifacts such as `.pdm-build/` and `__pypackages__/`, plus virtualenv and package-manager cache directories at any tree depth, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth Python files such as `pyproject.toml`, `setup.py`, `setup.cfg`, `requirements*.txt`, `Pipfile`, `Pipfile.lock`, `poetry.lock`, `uv.lock`, Python source files, or Python docs as local cache/build or virtualenv/dependency artifacts by filename alone.

The Python benchmark artifact guard rejects tracked local pytest-benchmark output such as `.benchmarks/`, with matching root and nested `.gitignore` coverage for local Python benchmark artifacts. It does not treat curated benchmark evidence under `docs/benchmarks/` or source helpers such as `scripts/bench_backend.py` as local benchmark artifacts.

The ML experiment/tracking artifact guard rejects tracked local W&B, MLflow, Lightning, and TensorBoard run outputs such as `.wandb/`, `wandb/`, `mlruns/`, `lightning_logs/`, and `events.out.tfevents.*`, with matching root and nested `.gitignore` coverage.

The Docker/container artifact guard rejects tracked local container runtime and override artifacts such as top-level `.docker/`, `docker-data/`, `docker-volumes/`, `docker-compose.override.yml`, `docker-compose.override.yaml`, `compose.override.yml`, and `compose.override.yaml`, with matching root `.gitignore` coverage.

The deployment platform artifact guard rejects tracked local Vercel and Netlify state directories such as `.vercel/` and `.netlify/` at any tree depth, plus Cloudflare Workers local state and secret/config files such as `.wrangler/`, `.dev.vars`, and `.dev.vars.*`, with matching root and nested `.gitignore` coverage. It does not treat source-of-truth deployment config such as `vercel.json`, `netlify.toml`, `wrangler.toml`, or `.dev.vars.example` as local platform state.

The Terraform/OpenTofu/Terragrunt infrastructure state artifact guard rejects tracked local infrastructure state artifacts such as `.terraform/`, `.tofu/`, `.terragrunt-cache/`, `.terraform.lock.hcl`, `.tofu.lock.hcl`, `.tfstate`, `.tfvars`, `.tfvars.json`, and `.tfplan` files, with matching root and nested `.gitignore` coverage for local infrastructure state artifacts.

The Nix build result artifact guard rejects tracked local result symlinks/directories such as root `result` and `result-*` outputs, with matching root `.gitignore` coverage for local Nix build result artifacts. It does not treat source-of-truth Nix files such as `flake.nix`, `flake.lock`, `default.nix`, `shell.nix`, or `*.nix` files as local build result artifacts.

The backend acceptance artifact summaries reject local paths, secret markers, and request/payload text in shareable `summary.json` and `summary.md`; keep full JSON artifacts and logs on the manual review path before publishing.

The browser-test artifact guard also blocks Cypress screenshot, video, and download output directories in addition to Playwright report directories, with matching root and nested `.gitignore` coverage for Cypress outputs.

The test report artifact guard also blocks Mochawesome report outputs such as `mochawesome-report/`, `mochawesome.html`, and `mochawesome.json` as local test report artifacts, with matching root and nested `.gitignore` coverage.

The test report artifact guard also blocks Cucumber/BDD report outputs such as `cucumber-report/`, `cucumber-reports/`, `cucumber-report.html`, `cucumber-report.ndjson`, `cucumber.json`, `cucumber-report.json`, `cucumber-report.xml`, and `cucumber.xml` as local test report artifacts, with matching root and nested `.gitignore` coverage where directory-shaped.

The test report artifact guard also blocks standalone xUnit XML report outputs such as `xunit.xml` and `*.xunit.xml` as local test report artifacts, with matching root `.gitignore` coverage.

The test report artifact guard also blocks Allure report/result directories as local test report artifacts, with matching root and nested `.gitignore` coverage for `allure-report/` and `allure-results/`.

The test report artifact guard also blocks Maven Surefire/Failsafe report directories as local test report artifacts, with matching root and nested `.gitignore` coverage for `surefire-reports/` and `failsafe-reports/`.

The test report artifact guard also blocks JaCoCo coverage outputs as local test report artifacts, with matching root and nested `.gitignore` coverage for `jacocoHtml/` and root `.gitignore` coverage for `jacoco.exec`, `jacoco.xml`, and `jacoco.csv`.

The test report artifact guard also blocks Cobertura coverage XML output (`cobertura.xml`) as a local test report artifact, with matching root `.gitignore` coverage.

The test report artifact guard also blocks cargo-tarpaulin coverage report outputs (`tarpaulin-report.html` and `tarpaulin-report.json`) as local test report artifacts, with matching root `.gitignore` coverage.

The test report artifact guard also blocks native compiler coverage outputs (`*.gcda`, `*.gcno`, and `*.gcov`) as local test report artifacts, with matching root `.gitignore` coverage.

The test report artifact guard also blocks mutation-testing report/cache outputs such as `.mutmut-cache/`, `.stryker-tmp/`, `mutation-report/`, `pit-reports/`, `mutation-report.html`, and `mutmut.sqlite`, with matching root and nested `.gitignore` coverage.

## What this launch currently proves

- The documented no-download public `/v1` contract routes and refusal envelopes work against the real backend process.
- Default CI stays offline with respect to model downloads, networked acceptance smoke, and non-default ONNX feature tests.
- Offline artifact QA covers optional public-contract smoke summaries, backend acceptance smoke success/failure summaries with share-safety checks, and MiniLM, SmolLM2, and Qwen2.5 optional API acceptance artifact schemas; the optional backend acceptance smoke itself remains networked and only produces current pinned-fixture evidence when downloads succeed.
- The current launch evidence snapshot is recorded in [`public-launch-evidence.md`](public-launch-evidence.md).

## What this launch does not prove

- It is not production readiness, performance capacity, model quality, or legal/license advice.
- It is not full OpenAI API parity; streaming chat and many OpenAI endpoints are unsupported.
- It does not prove external provider proxying; saved external entries are metadata placeholders and chat is refused.
- It does not prove general GGUF runtime, tokenizer execution, dequantization, generation, ONNX chat, PyTorch `.bin` loading, or arbitrary SafeTensors/Hugging Face execution.

## Troubleshooting

- **Missing tools:** install Rust (`cargo`), Node.js/npm, `curl`, and `python3`; Vite requires Node `20.19+` or `22.12+`.
- **Port conflicts:** stop old runs with `bash scripts/stop.sh`, or set `FATHOM_PORT` for backend-only runs. The contract smoke chooses a temporary local port automatically.
- **First build is slow:** `scripts/start.sh` and `scripts/start-backend.sh` build the Rust backend in release mode; the first run can take a few minutes.
- **Model download/network errors:** no-download gates should still pass without network. Catalog demos and `scripts/backend_acceptance_smoke.sh` need network access to fetch pinned fixtures.
- **Logs:** normal runs write backend logs under `~/.fathom/logs`; isolated smoke scripts print their temporary artifact/log locations while running and clean them unless configured to keep artifacts.
