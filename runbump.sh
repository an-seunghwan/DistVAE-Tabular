#!/bin/bash

# exit immediately if a command exits with a non-zero status
set -e

# Check if a version argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 [patch|minor|major]"
    exit 1
fi

VERSION_TYPE=$1

# Delete all remote tags to avoid conflicts
echo "Deleting all tags on remote..."
git push origin --delete $(git tag -l)

# Perform version bump
bumpversion $VERSION_TYPE

# Get the new version tag from bumpversion output
NEW_TAG=$(git describe --tags $(git rev-list --tags --max-count=1))

# Push only the new tag to remote
echo "Pushing the new tag $NEW_TAG to origin..."
git push origin $NEW_TAG
