# Fall 2025 Group 3

## Git workflow tips

If `git pull` reports `You are not currently on a branch`, it means the repository is in a detached HEAD state. To recover:

1. Inspect the current state:
   ```bash
   git status
   git branch -a
   ```
2. Switch back to the main working branch (this project uses a `work` branch; replace with `main` if that is your default):
   ```bash
   git switch work
   ```
   If the branch does not exist locally, fetch it and then switch:
   ```bash
   git fetch origin
   git switch work
   ```
3. Retry the pull or merge:
   ```bash
   git pull
   ```

When creating new feature branches, start from the shared branch:

```bash
git switch work
git pull
git switch -c feature/my-change
```

Commit locally, push your branch, and open a pull request as usual.
