#!/usr/bin/env bash
# Install the paper2code skill into an agent's skill directory (symlinked, so
# `git pull` updates it in place). Detects known skill directories automatically.
#
# Usage:
#   bash scripts/install_skill.sh              # install into the first detected dir
#   bash scripts/install_skill.sh --list       # show detected dirs and install state
#   bash scripts/install_skill.sh --dir PATH   # install into a specific dir
#   bash scripts/install_skill.sh --copy       # copy files instead of symlinking
#   bash scripts/install_skill.sh --remove     # uninstall (only removes links that
#                                              #   point back into this repo)
set -u
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ACTION="install"
TARGET_DIR=""
MODE="symlink"

while [ $# -gt 0 ]; do
  case "$1" in
    --list)   ACTION="list" ;;
    --dir)    TARGET_DIR="${2:?--dir needs a path}"; shift ;;
    --copy)   MODE="copy" ;;
    --remove) ACTION="remove" ;;
    *) echo "unknown option: $1"; exit 2 ;;
  esac
  shift
done

# Known skill roots (first existing one wins; ~/.zcode/skills is created if none exist)
CANDIDATES="$HOME/.zcode/skills $HOME/.claude/skills $HOME/.codex/skills $HOME/.agents/skills"
DETECTED=""
for d in $CANDIDATES; do
  if [ -d "$d" ]; then DETECTED="$DETECTED $d"; fi
done

if [ "$ACTION" = "list" ]; then
  echo "Detected skill roots:"
  for d in $DETECTED; do
    if [ -e "$d/paper2code/SKILL.md" ]; then state="installed"; else state="-"; fi
    echo "  $d  [$state]"
  done
  [ -z "$DETECTED" ] && echo "  (none found; install will create ~/.zcode/skills)"
  exit 0
fi

if [ -z "$TARGET_DIR" ]; then
  if [ -n "$DETECTED" ]; then
    TARGET_DIR=$(echo $DETECTED | awk '{print $1}')
  else
    TARGET_DIR="$HOME/.zcode/skills"
    echo "no skill directory found; creating $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
  fi
fi

SKILL_DIR="$TARGET_DIR/paper2code"
install_one() {  # $1 = source (relative), $2 = dest
  if [ "$MODE" = "copy" ]; then
    cp -R "$REPO_ROOT/$1" "$2"
  else
    ln -sfn "$REPO_ROOT/$1" "$2"
  fi
}

if [ "$ACTION" = "remove" ]; then
  for f in SKILL.md references scripts calibration; do
    p="$SKILL_DIR/$f"
    if [ -L "$p" ]; then
      dest=$(readlink "$p")
      case "$dest" in
        "$REPO_ROOT"/*) rm "$p" && echo "removed $p -> $dest" ;;
        *) echo "skip $p (not a link into this repo)" ;;
      esac
    elif [ -e "$p" ]; then
      echo "skip $p (not a symlink; remove manually if intended)"
    fi
  done
  [ -d "$SKILL_DIR" ] && rmdir "$SKILL_DIR" 2>/dev/null && echo "removed $SKILL_DIR"
  exit 0
fi

echo "installing paper2code -> $SKILL_DIR ($MODE)"
mkdir -p "$SKILL_DIR"
install_one "SKILL.md"    "$SKILL_DIR/SKILL.md"
install_one "references"  "$SKILL_DIR/references"
install_one "scripts"     "$SKILL_DIR/scripts"
install_one "calibration" "$SKILL_DIR/calibration"
echo "done. trigger it by asking your agent: 'reproduce this paper: <arXiv link>'"
echo "  (after any SKILL.md change, rerun: bash scripts/run_calibration.sh)"
