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

local entry_path = normalize(arg[0] or "")
if entry_path ~= "" and not entry_path:match("^/") and not entry_path:match("^%a:/") then
  entry_path = normalize(lfs.currentdir() .. "/" .. entry_path)
end

local support_dir = dirname(dirname(entry_path))

package.path = table.concat({
  support_dir .. "/lib/?.lua",
  support_dir .. "/lib/?/init.lua",
  package.path,
}, ";")

local cli = require("ntua_beamer_cli.cli")

local ok, err = pcall(function()
  cli.main({
    argv = arg,
    cwd = lfs.currentdir(),
    support_dir = support_dir,
  })
end)

if not ok then
  local message = tostring(err):gsub("^.-:%d+: ", "")
  io.stderr:write("Error: " .. message .. "\n")
  os.exit(1)
end
