#!/usr/bin/env bash
set -euxo pipefail

sed -i "s/^version =.*/version = $2/" setup.cfg
