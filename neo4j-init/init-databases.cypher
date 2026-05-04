// Creates vertical databases on first container initialization
// Safe to re-run: IF NOT EXISTS makes it idempotent
CREATE DATABASE meetings IF NOT EXISTS;
CREATE DATABASE projects IF NOT EXISTS;
CREATE DATABASE research IF NOT EXISTS;
