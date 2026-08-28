# Security policy

## Supported versions

Until the first public tag, only the current `main` branch receives security
fixes. After that, the newest tagged release and `main` are supported. Older
alpha releases may be asked to update before a report is investigated.

## Report a vulnerability

Use GitHub's private vulnerability reporting for this repository. Do not open a
public issue for a suspected vulnerability and do not attach private data,
credentials, voice samples, transcripts, or browser profiles to an issue.

Include the affected commit or version, impact, required preconditions, and a
minimal reproduction using synthetic data. Reports involving the installer,
authentication, approval signatures, path confinement, process identity, SSRF,
browser isolation, encrypted evidence, or secret exposure are in scope.

You should receive an acknowledgement within seven days. There is currently no
paid bug bounty and no promise of a specific remediation date.

## Security boundaries

Friday is designed for one trusted local operating-system user. It is not a
multi-tenant service. Same-user processes, the Linux user session, the kernel,
GPU drivers, systemd user services, and desktop keyring remain trusted parts of
the boundary.

The default installer exposes only loopback HTTPS. Do not bind Friday to a LAN
or public interface unless you have reviewed the controller, TLS, firewall, and
origin configuration for that deployment. See [Architecture](docs/architecture.md)
and [Privacy](docs/privacy.md) for the documented limits.
