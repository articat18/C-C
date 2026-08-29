# Gemini Agent Setup with Local Authentication

This guide sets up the project's Gemini agent on a developer workstation using Google Cloud
Application Default Credentials (ADC). It uses Vertex AI and your Google user identity—**not an API
key or a downloaded service-account key**.

## What You Need from the Project Owner

Before starting, confirm that you have:

- The Google Cloud **project ID** (not the project name or project number).
- A Gemini **model ID** supported by the project's Vertex AI location.
- Billing enabled on the project.
- The Vertex AI API (`aiplatform.googleapis.com`) enabled.
- The Vertex AI User role (`roles/aiplatform.user`) or equivalent permissions on the project.

An administrator may also need to grant Service Usage Consumer
(`roles/serviceusage.serviceUsageConsumer`) if your account cannot use the project for quota and
billing. Enabling APIs itself normally requires Service Usage Admin
(`roles/serviceusage.serviceUsageAdmin`); teammates do not need that role after the API is enabled.

## 1. Install the Local Tools

Install:

- Python 3.12 (recommended; Python 3.9+ is supported by the starter kit).
- The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install).
- Git.

Confirm that the commands are available:

```bash
python3.12 --version
gcloud --version
```

## 2. Create the Python Environment

Run these commands from `kuairand-starter-kit`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-agent.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

The dependency versions are pinned in `requirements-agent.txt`, including the Google Gen AI SDK and
Google Agent Development Kit.

## 3. Authenticate Locally with ADC

Set the Google Cloud CLI's active project, replacing `YOUR_PROJECT_ID`:

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

Then create local Application Default Credentials:

```bash
gcloud auth application-default login
```

Complete the browser sign-in with the same work account that has access to the project. The credentials
used by `gcloud` commands and the ADC credentials used by Python libraries are separate; running only
`gcloud auth login` is therefore not enough.

For a machine that cannot launch a browser, use:

```bash
gcloud auth application-default login --no-launch-browser
```

If prompted about a quota project, use the shared project. You can set it explicitly after login:

```bash
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

## 4. Configure the Agent

Create an untracked local configuration file:

```bash
cp .env.example .env
```

Edit `.env` and replace all placeholders:

```dotenv
GOOGLE_GENAI_USE_VERTEXAI=true
GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
GOOGLE_CLOUD_LOCATION=global
VERTEX_MODEL=YOUR_SUPPORTED_MODEL_ID
```

Use the project team's agreed model ID. Do not guess one: model availability depends on the project,
region, and enabled capabilities. The model must support function calling for later agent phases.

The `.env` file contains configuration rather than credentials, but it is intentionally ignored by
Git. ADC stores authentication separately in the Google Cloud CLI's local credential store. Do not add
API keys, access tokens, refresh tokens, or service-account JSON to `.env`.

## 5. Verify the Setup

First validate the Python packages and environment variables without calling Gemini or incurring model
charges:

```bash
python -m agent.vertex_healthcheck
```

Expected output resembles:

```text
Vertex configuration valid: project=YOUR_PROJECT_ID location=global model=YOUR_SUPPORTED_MODEL_ID
configuration-only check passed; no API request was made
```

Then make one small live request. This call uses Vertex AI quota and may incur a charge:

```bash
python -m agent.vertex_healthcheck --live
```

The final line should be:

```text
live Vertex AI health check passed: VERTEX_OK
```

You can also run the configuration tests:

```bash
python -m unittest tests.test_config
```

## Troubleshooting

### `gcloud: command not found`

Install the Google Cloud CLI, restart the shell, and run `gcloud --version` again.

### `google-genai is not installed`

Activate `.venv` and reinstall the pinned dependencies:

```bash
source .venv/bin/activate
python -m pip install -r requirements-agent.txt
```

### `missing Vertex AI configuration`

Run the command from the `kuairand-starter-kit` directory so `python-dotenv` can find `.env`. Confirm
that every placeholder in `.env` was replaced.

### Authentication errors or `DefaultCredentialsError`

Create ADC again with the correct account:

```bash
gcloud auth application-default login
```

If `GOOGLE_APPLICATION_CREDENTIALS` points to an old JSON file, it overrides the local ADC login. Unless
your project owner explicitly requires that file, remove it from the current shell and retry:

```bash
unset GOOGLE_APPLICATION_CREDENTIALS
```

### `403 PermissionDenied`

Ask the project owner to verify that:

- Your signed-in account has Vertex AI User (`roles/aiplatform.user`) or equivalent permissions.
- The Vertex AI API is enabled in the same project specified by `GOOGLE_CLOUD_PROJECT`.
- Billing is enabled.
- Organization policies allow the selected model and location.

### Quota-project or `serviceusage.services.use` error

Ask an administrator for Service Usage Consumer (`roles/serviceusage.serviceUsageConsumer`) on the
project, then run:

```bash
gcloud auth application-default set-quota-project YOUR_PROJECT_ID
```

### Model not found or unsupported location

Confirm `VERTEX_MODEL` and `GOOGLE_CLOUD_LOCATION` with the project owner. A valid model ID in one
location may not be available in another.

## Security and Sign-out

- Never commit `.env`, credential JSON files, access tokens, or copied ADC files.
- Never send local ADC files to another teammate; everyone should authenticate with their own identity.
- Prefer user ADC or approved service-account impersonation over downloadable service-account keys.
- Revoke the local ADC credentials when they are no longer needed:

```bash
gcloud auth application-default revoke
```

Revoking ADC does not necessarily sign the Google Cloud CLI out. To revoke the CLI account too, run
`gcloud auth revoke` separately.

## References

- [Gemini API in Vertex AI quickstart](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/start/quickstart)
- [Set up ADC for local development](https://docs.cloud.google.com/docs/authentication/set-up-adc-local-dev-environment)
- [`gcloud auth application-default login` reference](https://docs.cloud.google.com/sdk/gcloud/reference/auth/application-default/login)
- [Set the ADC quota project](https://docs.cloud.google.com/sdk/gcloud/reference/auth/application-default/set-quota-project)
