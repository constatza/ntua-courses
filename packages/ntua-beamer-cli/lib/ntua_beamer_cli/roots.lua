local fs = require("ntua_support.fs")
local path = require("ntua_support.path")

local M = {}

local ignored_dirs = {
  [".build"] = true,
  ["dist"] = true,
  ["build"] = true,
  ["out"] = true,
  ["target"] = true,
  ["node_modules"] = true,
  ["vendor"] = true,
  ["venv"] = true,
  [".venv"] = true,
  ["__pycache__"] = true,
}

local function is_hidden(name)
  return name:sub(1, 1) == "."
end

local function should_skip_dir(_, entry)
  return is_hidden(entry) or ignored_dirs[entry] or false
end

local function is_root_tex(target)
  if path.extension(target) ~= ".tex" then
    return false
  end

  local content = fs.read_file(target)
  return content:find("\\documentclass", 1, true) ~= nil
end

local function choose_candidate(candidates, owner_dir)
  if #candidates == 0 then
    return nil
  end

  if #candidates == 1 then
    return candidates[1]
  end

  for _, candidate in ipairs(candidates) do
    if path.basename(candidate) == "main.tex" then
      return candidate
    end
  end

  local relative = {}
  for _, candidate in ipairs(candidates) do
    table.insert(relative, path.relpath(candidate, owner_dir))
  end
  table.sort(relative)

  error("multiple root .tex files found in " .. owner_dir .. ": " .. table.concat(relative, ", "))
end

local function roots_in_dir(dir)
  local candidates = {}
  for _, entry in ipairs(fs.list_dir_names(dir)) do
    local target = path.join(dir, entry)
    if fs.is_file(target) and not is_hidden(entry) and is_root_tex(target) then
      table.insert(candidates, target)
    end
  end
  table.sort(candidates)
  return candidates
end

function M.resolve_one(target, cwd)
  if not target or target == "" then
    local current = path.abspath(cwd, cwd)
    while true do
      local match = choose_candidate(roots_in_dir(current), current)
      if match then
        return match
      end

      local parent = path.dirname(current)
      if parent == current then
        break
      end
      current = parent
    end

    error("no root .tex file found from " .. path.abspath(cwd, cwd) .. " upward")
  end

  local absolute = path.abspath(target, cwd)
  if not fs.exists(absolute) then
    error("path was not found: " .. absolute)
  end

  if fs.is_file(absolute) and not is_root_tex(absolute) then
    error("file is not a root LaTeX document: " .. absolute)
  end

  if fs.is_file(absolute) then
    return absolute
  end

  if not fs.is_dir(absolute) then
    error("path was not found: " .. absolute)
  end

  local match = choose_candidate(roots_in_dir(absolute), absolute)
  if match then
    return match
  end

  error("no root .tex file found in " .. absolute)
end

function M.resolve_all(target, cwd)
  local start_dir = target and path.abspath(target, cwd) or path.abspath(cwd, cwd)
  if fs.is_file(start_dir) then
    error("--all expects a directory, not a file")
  end

  if not fs.is_dir(start_dir) then
    error("directory was not found: " .. start_dir)
  end

  local found = {}
  fs.walk_files(start_dir, function(file)
    if is_root_tex(file) then
      table.insert(found, file)
    end
  end, should_skip_dir)

  table.sort(found)

  if #found == 0 then
    error("no root .tex files were found under " .. start_dir)
  end

  return found
end

return M
