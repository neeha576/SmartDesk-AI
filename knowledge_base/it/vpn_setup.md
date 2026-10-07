---
doc_id: IT-002
title: VPN Setup and Troubleshooting
domain: IT
category: IT Support
owner: IT Network Team
last_updated: 2026-06-15
---

# VPN Setup and Troubleshooting

NMTech uses **GlobalProtect** VPN. You need the VPN to reach internal systems (Jira, the code repository, the finance and HR databases) when you are not in the office. Email, NovaHR and Microsoft Teams work without the VPN.

## Supported clients
| Operating system | Client | Where to get it |
|---|---|---|
| Windows 10/11 | GlobalProtect 6.x | Software Center (pre-installed on company laptops) |
| macOS 13 or later | GlobalProtect 6.x | Self Service app |
| Ubuntu 22.04 / 24.04 | GlobalProtect for Linux | IT Service Portal → Request Software |

Personal devices cannot connect to the VPN.

## Connecting
1. Open GlobalProtect from the system tray (Windows) or menu bar (Mac).
2. Enter the portal address **vpn.nmtech.com** (first time only).
3. Sign in with your NMTech email and password.
4. Approve the MFA prompt in Microsoft Authenticator.
5. The icon shows **Connected**. Sessions last up to 12 hours before you must sign in again.

## Common errors
- **"Gateway not responding"** – check your home internet first, then disconnect and reconnect. If it persists, switch from Wi-Fi to a mobile hotspot to rule out your router.
- **"Authentication failed"** – your password may have expired. Reset it at the Self-Service Password Portal, then try again.
- **No MFA prompt arrives** – open Microsoft Authenticator manually and pull down to refresh. Check that your phone's time is set automatically.
- **"Client version not supported"** – update GlobalProtect from Software Center or Self Service.
- **Connected but internal sites won't load** – disconnect, flush DNS (Windows: `ipconfig /flushdns`), and reconnect.

## Usage rules
- Do not leave the VPN connected on public Wi-Fi without the laptop locked.
- Streaming and large personal downloads over the VPN are not allowed.

## Still stuck?
Raise a ticket in the IT Service Portal or call ext. 2020. Include the exact error message and a screenshot.
