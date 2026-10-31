local fs = require("ntua_support.fs")
local path = require("ntua_support.path")
local process = require("ntua_support.process")

local packaging = nil

local M = {}

local function manifest_path(support_dir)
  return path.join(support_dir, "install-manifest.txt")
end

local function home_dir()
  return path.normalize(os.getenv(path.is_windows() and "USERPROFILE" or "HOME") or "")
end

local function local_appdata_dir()
  return path.normalize(os.getenv("LOCALAPPDATA") or "")
end

local function local_share_dir()
  local xdg = path.normalize(os.getenv("XDG_DATA_HOME") or "")
  if xdg ~= "" then
    return xdg
  end

  return path.join(home_dir(), ".local", "share")
end

local function path_entries()
  local entries = {}
  local raw = os.getenv("PATH") or ""

  for part in string.gmatch(raw .. path.path_sep(), "(.-)" .. path.path_sep()) do
    if part ~= "" then
      table.insert(entries, path.normalize(part))
    end
  end

  return entries
end

local function is_path_entry(target)
  local comparable = path.comparable(target)
  for _, entry in ipairs(path_entries()) do
    if path.comparable(entry) == comparable then
      return true
    end
  end

  return false
end

local function choose_bin_dir(explicit_dir)
  if explicit_dir and explicit_dir ~= "" then
    return path.normalize(explicit_dir)
  end

  if path.is_windows() then
    return path.join(local_appdata_dir(), "Programs", "ntua-beamer", "bin")
  end

  return path.join(home_dir(), ".local", "bin")
end

local function choose_support_dir(explicit_dir)
  if explicit_dir and explicit_dir ~= "" then
    return path.normalize(explicit_dir)
  end

  if path.is_windows() then
    return path.join(local_appdata_dir(), "ntua-beamer")
  end

  return path.join(local_share_dir(), "ntua-beamer")
end

local function unix_launcher(entry_path)
  return [[#!/bin/sh
set -eu
exec texlua "]] .. entry_path .. [[" "$@"
]]
end

local function windows_launcher(entry_path)
  return [[@echo off
setlocal
texlua "]] .. entry_path:gsub("/", "\\") .. [[" %*
exit /b %ERRORLEVEL%
]]
end

local function launcher_paths(bin_dir, support_dir)
  local entries_dir = path.join(support_dir, "entry")
  return {
    workflow_bin = path.is_windows() and path.join(bin_dir, "ntua-beamer.cmd") or path.join(bin_dir, "ntua-beamer"),
    class_bin = path.is_windows() and path.join(bin_dir, "ntuabeamer-class.cmd") or path.join(bin_dir, "ntuabeamer-class"),
    workflow_entry = path.join(entries_dir, "ntua-beamer.lua"),
    class_entry = path.join(entries_dir, "ntuabeamer-class.lua"),
  }
end

local function write_manifest(support_dir, targets)
  local lines = {
    "workflow_bin=" .. targets.workflow_bin,
    "class_bin=" .. targets.class_bin,
  }

  fs.write_file(manifest_path(support_dir), table.concat(lines, "\n") .. "\n")
end

local function read_manifest(support_dir)
  local target = manifest_path(support_dir)
  if not fs.is_file(target) then
    return nil
  end

  local result = {}
  local content = fs.read_file(target)
  for line in content:gmatch("[^\r\n]+") do
    local key, value = line:match("^([%w_]+)=(.+)$")
    if key and value then
      result[key] = value
    end
  end

  return result
end

local function launcher_matches_support(target, support_dir)
  if not fs.is_file(target) then
    return false
  end

  local content = fs.read_file(target)
  if content == "" then
    return false
  end

  local normalized = content:gsub("\\", "/")
  local workflow_entry = path.join(support_dir, "entry", "ntua-beamer.lua")
  local class_entry = path.join(support_dir, "entry", "ntuabeamer-class.lua")

  if normalized:find(workflow_entry, 1, true) ~= nil then
    return true
  end

  return normalized:find(class_entry, 1, true) ~= nil
end

local function legacy_launcher_candidates(support_dir)
  local names = path.is_windows()
      and {"ntua-beamer.cmd", "ntuabeamer-class.cmd"}
      or {"ntua-beamer", "ntuabeamer-class"}

  local candidates = {}
  for _, entry in ipairs(path_entries()) do
    for _, name in ipairs(names) do
      local target = path.join(entry, name)
      if launcher_matches_support(target, support_dir) then
        table.insert(candidates, target)
      end
    end
  end

  return candidates
end

local function unique_paths(values)
  local seen = {}
  local result = {}

  for _, value in ipairs(values) do
    local comparable = path.comparable(value)
    if value ~= "" and not seen[comparable] then
      seen[comparable] = true
      table.insert(result, value)
    end
  end

  return result
end

local function install_launchers(bin_dir, support_dir)
  local targets = launcher_paths(bin_dir, support_dir)

  fs.ensure_dir(bin_dir)
  if path.is_windows() then
    fs.write_file(targets.workflow_bin, windows_launcher(targets.workflow_entry))
    fs.write_file(targets.class_bin, windows_launcher(targets.class_entry))
    return targets
  end

  fs.write_file(targets.workflow_bin, unix_launcher(targets.workflow_entry))
  fs.write_file(targets.class_bin, unix_launcher(targets.class_entry))
  process.run_command({"chmod", "+x", targets.workflow_bin}, {quiet = true})
  process.run_command({"chmod", "+x", targets.class_bin}, {quiet = true})
  return targets
end

local function update_windows_user_path(bin_dir)
  if is_path_entry(bin_dir) then
    return true
  end

  local current = os.getenv("PATH") or ""
  local new_path = current == "" and bin_dir or (current .. ";" .. bin_dir)
  local shell = process.command_exists("powershell") and "powershell" or (process.command_exists("pwsh") and "pwsh" or nil)
  if not shell then
    return false
  end

  local command = string.format("[Environment]::SetEnvironmentVariable('Path', %q, 'User')", new_path)
  return process.run_command({shell, "-NoProfile", "-Command", command}, {quiet = true})
end

local function remove_if_exists(target)
  if fs.exists(target) then
    fs.remove_tree(target)
  end
end

local function install_payload(context, support_dir)
  packaging = packaging or require("ntuabeamer_class.packaging")

  remove_if_exists(support_dir)

  fs.ensure_dir(path.join(support_dir, "lib"))
  fs.ensure_dir(path.join(support_dir, "entry"))
  fs.ensure_dir(path.join(support_dir, "templates"))
  fs.ensure_dir(path.join(support_dir, "framework"))

  fs.copy_tree(path.join(context.repo_root, "packages", "shared", "lib"), path.join(support_dir, "lib"))
  fs.copy_tree(path.join(context.repo_root, "packages", "ntua-beamer-cli", "lib"), path.join(support_dir, "lib"))
  fs.copy_tree(path.join(context.repo_root, "packages", "ntuabeamer-class", "lib"), path.join(support_dir, "lib"))
  fs.copy_file(path.join(context.repo_root, "packages", "ntua-beamer-cli", "entry.lua"), path.join(support_dir, "entry", "ntua-beamer.lua"))
  fs.copy_file(path.join(context.repo_root, "packages", "ntuabeamer-class", "entry.lua"), path.join(support_dir, "entry", "ntuabeamer-class.lua"))
  fs.copy_tree(path.join(context.repo_root, "templates"), path.join(support_dir, "templates"))
  fs.copy_tree(path.join(context.repo_root, "packages", "ntuabeamer-class", "framework", "assets"), path.join(support_dir, "framework", "assets"))
  fs.copy_tree(path.join(context.repo_root, "packages", "ntuabeamer-class", "framework"), path.join(support_dir, "framework", "source"))
  fs.ensure_dir(path.join(support_dir, "framework", "standalone"))
  fs.write_file(
    path.join(support_dir, "framework", "standalone", "ntuabeamer.cls"),
    packaging.build_standalone_class(path.join(context.repo_root, "packages", "ntuabeamer-class", "framework"))
  )
end

local function uninstall_targets(bin_dir, support_dir)
  local manifest = read_manifest(support_dir)
  if manifest then
    return unique_paths({
      manifest.workflow_bin or "",
      manifest.class_bin or "",
    })
  end

  local targets = launcher_paths(bin_dir, support_dir)
  local result = {
    targets.workflow_bin,
    targets.class_bin,
  }

  for _, target in ipairs(legacy_launcher_candidates(support_dir)) do
    table.insert(result, target)
  end

  return unique_paths(result)
end

local function parse_args(arguments)
  local options = {
    uninstall = false,
    bin_dir = nil,
    support_dir = nil,
  }

  local index = 1
  while index <= #arguments do
    local value = arguments[index]
    if value == "setup" then
      -- ignore
    elseif value == "--uninstall" then
      options.uninstall = true
    elseif value == "--bin-dir" then
      index = index + 1
      if index > #arguments then
        error("missing value for --bin-dir")
      end
      options.bin_dir = arguments[index]
    elseif value == "--support-dir" then
      index = index + 1
      if index > #arguments then
        error("missing value for --support-dir")
      end
      options.support_dir = arguments[index]
    else
      error("unknown setup option '" .. tostring(value) .. "'")
    end
    index = index + 1
  end

  return options
end

local function usage()
  print([[
Usage:
  texlua scripts/ntua-beamer.lua setup
  texlua scripts/ntua-beamer.lua setup --uninstall

Options:
  --bin-dir PATH       Install the launchers into a specific directory.
  --support-dir PATH   Install the support payload into a specific directory.
]])
end

function M.main(context)
  local command = context.argv[1]
  if not command or command == "--help" or command == "-h" or command == "help" then
    usage()
    return
  end

  if command ~= "setup" then
    error("only setup is supported from the repo bootstrap script")
  end

  local options = parse_args(context.argv)
  local bin_dir = choose_bin_dir(options.bin_dir)
  local support_dir = choose_support_dir(options.support_dir)

  if options.uninstall then
    for _, target in ipairs(uninstall_targets(bin_dir, support_dir)) do
      remove_if_exists(target)
    end
    remove_if_exists(support_dir)
    print("Removed launchers and support payload.")
    return
  end

  install_payload(context, support_dir)
  local installed = install_launchers(bin_dir, support_dir)
  write_manifest(support_dir, installed)

  print("Installed workflow launcher -> " .. installed.workflow_bin)
  print("Installed class launcher -> " .. installed.class_bin)
  print("Installed support payload -> " .. support_dir)

  if not path.is_windows() then
    if is_path_entry(bin_dir) then
      print("Launcher directory is already on PATH.")
      return
    end

    print("Add this line to your shell profile, then restart your terminal:")
    print('  export PATH="' .. bin_dir .. ':$PATH"')
    return
  end

  if is_path_entry(bin_dir) then
    print("Launcher directory is already on PATH.")
    return
  end

  if update_windows_user_path(bin_dir) then
    print("Added launcher directory to the user PATH. Restart your terminal to use the tools.")
    return
  end

  print("Add this directory to your user PATH, then restart your terminal:")
  print("  " .. bin_dir)
end

return M
