-- PostgreSQL schema for Outfit Selector AI (standalone, no Supabase)
-- Run once: Docker mounts this into /docker-entrypoint-initdb.d/

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email         TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL,
  name          TEXT NOT NULL DEFAULT 'User',
  avatar_url    TEXT,
  is_premium    BOOLEAN DEFAULT FALSE,
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wardrobe_items (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
  category        TEXT,
  type            TEXT,
  color           TEXT,
  hex             TEXT,
  pattern         TEXT,
  style           TEXT,
  season          TEXT,
  confidence      REAL,
  image_path      TEXT,
  cutout_path     TEXT,
  embedding       REAL[],
  secondary_color TEXT,
  secondary_hex   TEXT,
  created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS saved_outfits (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  outfit_data JSONB NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS wear_logs (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  worn_on     DATE NOT NULL,
  outfit_data JSONB NOT NULL,
  created_at  TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE (user_id, worn_on)
);

CREATE TABLE IF NOT EXISTS outfit_plans (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id    UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  kind       TEXT NOT NULL CHECK (kind IN ('week', 'trip')),
  title      TEXT,
  starts_on  DATE NOT NULL,
  days       JSONB NOT NULL,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
