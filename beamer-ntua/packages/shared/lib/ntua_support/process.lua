local lfs = require("lfs")

local path = require("ntua_support.path")

local M = {}

local function trim(value)
  return (value or ""):gsub("^%s+", ""):gsub("%s+$", "")
end

local function shell_quote(value)
  value = tostring(value)
  value = value:gsub('"', '\\"')
  return '"' .. value .. '"'
end

local function normalize_status(ok, _, code)
  if type(ok) == "number" then
    return ok == 0, ok
  end
  if ok == true then
    return true, code or 0
  end
  return false, code or 1
end

local function with_cwd(cwd, fn)
  if not cwd or cwd == "" then
    return fn()
  end

  local current = lfs.currentdir()
  assert(lfs.chdir(cwd))

  local result = {pcall(fn)}
  lfs.chdir(current)

  if not result[1] then
    error(result[2])
  end

  return table.unpack(result, 2)
end

function M.run_shell(command, options)
  options = options or {}

  if not options.quiet then
    print("> " .. command)
  end

  return with_cwd(options.cwd, function()
    local ok, why, code = os.execute(command)
    return normalize_status(ok, why, code)
  end)
end

function M.capture_shell(command, options)
  options = options or {}

  return with_cwd(options.cwd, function()
    local pipe = io.popen(command .. " 2>&1")
    if not pipe then
      return "", false, 1
    end

    local output = pipe:read("*a")
    local ok, why, code = pipe:close()
    local success, exit_code = normalize_status(ok, why, code)
    return trim(output), success, exit_code
  end)
end

function M.run_command(args, options)
  local parts = {}
  for _, value in ipairs(args) do
    table.insert(parts, shell_quote(value))
  end
  return M.run_shell(table.concat(parts, " "), options)
end

function M.capture_command(args, options)
  local parts = {}
  for _, value in ipairs(args) do
    table.insert(parts, shell_quote(value))
  end
  return M.capture_shell(table.concat(parts, " "), options)
end

function M.command_exists(name)
  local probe
  if path.is_windows() then
    probe = "where " .. shell_quote(name)
  else
    probe = "command -v " .. shell_quote(name)
  end

  local output, ok = M.capture_shell(probe)
  return ok and output ~= ""
end

return M
