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

If `git pull --rebase` stops with `cannot pull with rebase: You have unstaged changes`, clean up your working tree before retrying:

1. Check what is modified:
   ```bash
   git status
   ```
2. Either commit the work in progress or stash it temporarily:
   ```bash
   git add <files>
   git commit -m "Save work in progress"
   # or stash instead of committing
   git stash push -u -m "wip before pull"
   ```
3. Re-run the pull (with rebase if desired):
   ```bash
   git pull --rebase
   ```
4. If you used `git stash`, restore the saved changes afterwards:
   ```bash
  git stash pop
  ```

If your IDE shows a rebase in progress that you did not intend, you can safely return to the previous state:

1. Inspect the rebase status and current branch:
   ```bash
   git status
   ```
2. Abort the in-progress rebase to restore the branch to its last committed state:
   ```bash
   git rebase --abort
   ```
3. Ensure you are on the expected branch (replace `work` with `main` if that is your default):
   ```bash
   git switch work
   ```
4. Fetch and pull the latest changes normally:
   ```bash
   git fetch origin
   git pull
   ```
