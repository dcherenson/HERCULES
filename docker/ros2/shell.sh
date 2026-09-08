#!/usr/bin/env bash
source "$(dirname -- "$0")/common.sh"
ensure_dev
dev_exec bash "$@"
