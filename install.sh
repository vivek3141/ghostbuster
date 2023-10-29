#!/bin/bash
set -euo pipefail

if [[ -z $VIRTUAL_ENV ]]; then
    echo "Don't run this outside a virtualenv"; exit 1;
fi

pip install -r requirements.txt -c constraints.txt
pip install -e .

echo
echo "Completed successfully."
