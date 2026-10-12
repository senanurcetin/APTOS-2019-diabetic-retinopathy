# Security policy

## Reporting a vulnerability

Please disclose privately by emailing [senanur.cetin.work@gmail.com](mailto:senanur.cetin.work@gmail.com) with the subject `[APTOS-2019-diabetic-retinopathy security]`. Include the affected version or commit, impact, and minimal safe reproduction steps. Do **not** disclose credentials, personal data, exploits against third-party systems, or sensitive logs in public GitHub issues.

This is a portfolio/research project with no guaranteed response time or supported security-release schedule.

## Use and deployment boundaries

This is a medical-image research project, **not a medical device or diagnostic tool**. Do not submit identifiable retinal scans or other patient information to a public demo. Clinical use requires independently validated performance, privacy controls, regulatory review and qualified clinical oversight.

Use environment variables or a dedicated secret manager for credentials. Do not commit .env files, data containing personal information, or private model/data artifacts.
