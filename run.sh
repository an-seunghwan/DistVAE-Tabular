#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Check if a version argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [patch|minor|major]"
    exit 1
fi

VERSION_TYPE=$1

# Check if bumpversion is installed
if ! command -v bumpversion &> /dev/null; then
    echo "bumpversion not found. Installing..."
    pip install bumpversion
fi

# Automatically commit uncommitted changes
git add .
git commit -m "Auto-commit before version bump" || echo "No changes to commit"

# Step 1: Update the version
echo "Updating version ($VERSION_TYPE)..."
bumpversion $VERSION_TYPE

# Step 2: Push the new tag to origin
echo "Pushing tags to origin..."
git push origin --tags

echo "Version updated and tags pushed to origin successfully!"
