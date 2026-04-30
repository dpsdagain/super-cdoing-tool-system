# pylint: disable=too-many-branches,too-many-return-statements
import re
import shlex
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- Core Patterns from Claude Code ---
COMMAND_SUBSTITUTION_PATTERNS = [
    (r"<\(", "process substitution <()"),
    (r">\(", "process substitution >()"),
    (r"=\(", "Zsh process substitution =()"),
    (r"(?:^|[\s;&|])=[a-zA-Z_]", "Zsh equals expansion (=cmd)"),
    (r"\$\(", "$() command substitution"),
    (r"\$\{", "${} parameter substitution"),
    (r"\$\[", "$[] legacy arithmetic expansion"),
    (r"~\[", "Zsh-style parameter expansion"),
    (r"\(e:", "Zsh-style glob qualifiers"),
    (r"\(\+", "Zsh glob qualifier with command execution"),
    (r"\}\s*always\s*\{", "Zsh always block"),
    (r"<#", "PowerShell comment syntax"),
]

ZSH_DANGEROUS_COMMANDS = {
    "zmodload",
    "emulate",
    "sysopen",
    "sysread",
    "syswrite",
    "sysseek",
    "zpty",
    "ztcp",
    "zsocket",
    "mapfile",
    "zf_rm",
    "zf_mv",
    "zf_ln",
    "zf_chmod",
    "zf_chown",
    "zf_mkdir",
    "zf_rmdir",
    "zf_chgrp",
}

DANGEROUS_COMMANDS = {
    "rm",
    "mkfs",
    "dd",
    "chmod",
    "chown",
    "wget",
    "curl",
    "nc",
    "netcat",
    "socat",
    "telnet",
    "python",
    "python3",
    "perl",
    "ruby",
    "php",
    "node",
    "npm",
    "npx",
    "yarn",
    "sudo",
    "su",
    "doas",
    "docker",
    "kubectl",
    "k8s",
    "eval",
    "exec",
    "source",
    ".",
    "alias",
    "unalias",
    "bind",
    "trap",
    "kill",
    "killall",
    "pkill",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "init",
    "systemctl",
    "service",
    "journalctl",
    "dmesg",
    "modprobe",
    "insmod",
    "rmmod",
    "lsmod",
    "mount",
    "umount",
    "chroot",
    "pivot_root",
    "unshare",
    "nsenter",
    "iptables",
    "ufw",
    "firewalld",
    "ufw-default",
    "pf",
    "pfctl",
    "ip",
    "ifconfig",
    "route",
    "netstat",
    "ss",
    "arp",
    "iwconfig",
    "iw",
    "nmcli",
    "wpa_cli",
    "wpa_supplicant",
    "airmon-ng",
    "airodump-ng",
    "aireplay-ng",
    "aircrack-ng",
    "reaver",
    "bully",
    "pixiewps",
    "hashcat",
    "john",
    "hydra",
    "medusa",
    "nmap",
    "masscan",
    "zmap",
    "rustscan",
    "nikto",
    "sqlmap",
    "sqlping",
    "sqlninja",
}


class BashSecurityAnalyzer:
    @staticmethod
    def extract_quoted_content(command: str) -> Dict[str, str]:
        """Extracts content handling bash quotes correctly."""
        with_double_quotes = ""
        fully_unquoted = ""
        unquoted_keep_chars = ""
        in_single = False
        in_double = False
        escaped = False

        for char in command:
            if escaped:
                escaped = False
                if not in_single:
                    with_double_quotes += char
                if not in_single and not in_double:
                    fully_unquoted += char
                if not in_single and not in_double:
                    unquoted_keep_chars += char
                continue

            if char == "\\" and not in_single:
                escaped = True
                if not in_single:
                    with_double_quotes += char
                if not in_single and not in_double:
                    fully_unquoted += char
                if not in_single and not in_double:
                    unquoted_keep_chars += char
                continue

            if char == "'" and not in_double:
                in_single = not in_single
                unquoted_keep_chars += char
                continue

            if char == '"' and not in_single:
                in_double = not in_double
                unquoted_keep_chars += char
                continue

            if not in_single:
                with_double_quotes += char
            if not in_single and not in_double:
                fully_unquoted += char
            if not in_single and not in_double:
                unquoted_keep_chars += char

        return {
            "with_double_quotes": with_double_quotes,
            "fully_unquoted": fully_unquoted,
            "unquoted_keep_chars": unquoted_keep_chars,
        }

    @staticmethod
    def analyze(command: str) -> Dict[str, Any]:
        """
        Analyzes a bash command using a hybrid Regex/AST approach (ported from Claude Code).
        Returns a dict: {"allowed": bool, "reason": str}
        """
        if not command.strip():
            return {"allowed": True, "reason": "Empty command"}

        # 1. Base formatting checks
        if re.search(r"^\s*\t", command) or command.strip().startswith("-"):
            return {
                "allowed": False,
                "reason": "Command fragment (starts with tab or flag)",
            }

        # 2. Extract quotes
        quoted_data = BashSecurityAnalyzer.extract_quoted_content(command)
        unquoted = quoted_data["fully_unquoted"]

        # 3. Check for Dangerous Patterns (Command Substitution)
        for pattern, message in COMMAND_SUBSTITUTION_PATTERNS:
            if re.search(pattern, unquoted):
                return {
                    "allowed": False,
                    "reason": f"Dangerous pattern detected: {message}",
                }

        # 4. Check for Metacharacters that might break shlex or execute multiple commands
        # Even with shell=False, some commands might evaluate these if passed as args
        if re.search(r"[&|;><`]", unquoted):
            return {
                "allowed": False,
                "reason": "Shell metacharacters (&, |, ;, >, <, `) are not allowed.",
            }

        # 5. Parse using shlex to get base command and check against Denylists
        try:
            tokens = shlex.split(command)
            if not tokens:
                return {"allowed": True, "reason": ""}

            base_cmd = tokens[0].lower()

            if base_cmd in ZSH_DANGEROUS_COMMANDS:
                return {
                    "allowed": False,
                    "reason": f"Zsh-specific dangerous module execution: {base_cmd}",
                }

            if base_cmd in DANGEROUS_COMMANDS:
                return {
                    "allowed": False,
                    "reason": f"Command restricted by security policy: {base_cmd}",
                }

        except ValueError as e:
            return {
                "allowed": False,
                "reason": f"Malformed command string (unclosed quotes): {e}",
            }

        return {"allowed": True, "reason": "Command passed semantic analysis."}
