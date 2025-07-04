# Commit Command

Generate a concise commit message based on current changes and create a git commit.

## Usage
```
/commit
```

## What it does
1. Analyzes current git status and changes
2. Creates a commit message with:
   - Short, descriptive header (under 50 characters)
   - 3-7 bullet points summarizing key changes
   - Appropriate subheader if needed
3. Commits the changes with the generated message
4. Does NOT include Claude authorship attribution
5. Does NOT push to remote (manual push required)

## Example Output
```
feat: Add user authentication system

- Implement JWT token generation and validation
- Add password hashing with bcrypt
- Create login/logout endpoints
- Add middleware for protected routes
- Update user model with auth fields
```

The command will stage relevant files automatically and create the commit, but you'll need to manually push when ready.
