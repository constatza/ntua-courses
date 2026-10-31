local lfs = require("lfs")

local function normalize(value)
  if not value or value == "" then
    return ""
  end

  value = value:gsub("\\", "/")
  value = value:gsub("/+", "/")

  if #value > 1 and value:sub(-1) == "/" and not value:match("^%a:/$") then
    value = value:sub(1, -2)
  end

  return value
end

local function dirname(value)
  value = normalize(value)
  if value == "" or value == "." then
    return "."
  end
  if value == "/" or value:match("^%a:/$") then
    return value
  end
  return value:match("^(.*)/[^/]+$") or "."
end

local script_path = normalize(arg[0] or "")
if script_path ~= "" and not script_path:match("^/") and not script_path:match("^%a:/") then
  script_path = normalize(lfs.currentdir() .. "/" .. script_path)
end

local script_dir = dirname(script_path)
local repo_root = dirname(script_dir)

package.path = table.concat({
  repo_root .. "/packages/shared/lib/?.lua",
  repo_root .. "/packages/shared/lib/?/init.lua",
  repo_root .. "/packages/ntuabeamer-class/lib/?.lua",
  repo_root .. "/packages/ntuabeamer-class/lib/?/init.lua",
  repo_root .. "/scripts/?.lua",
  repo_root .. "/scripts/?/init.lua",
  package.path,
}, ";")

local setup = require("setup")

local ok, err = pcall(function()
  setup.main({
    argv = arg,
    repo_root = repo_root,
    cwd = lfs.currentdir(),
  })
end)

if not ok then
  local message = tostring(err):gsub("^.-:%d+: ", "")
  io.stderr:write("Error: " .. message .. "\n")
  os.exit(1)
end
