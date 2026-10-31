local process = require("ntua_support.process")

local M = {}

local function status(kind, message)
  print("[" .. kind .. "] " .. message)
end

local function check_required(name)
  if process.command_exists(name) then
    status("ok", name .. " found")
    return true
  end

  status("error", name .. " not found")
  return false
end

function M.run(context)
  local ok = true

  ok = check_required("texlua") and ok
  ok = check_required("latexmk") and ok

  for _, engine in ipairs({"pdflatex", "lualatex", "xelatex"}) do
    if process.command_exists(engine) then
      status("ok", engine .. " found")
    else
      status("warn", engine .. " not found")
    end
  end

  if process.command_exists("biber") then
    status("ok", "biber found")
  else
    status("warn", "biber not found; bibliography-enabled documents will not build")
  end

  if process.command_exists("kpsewhich") then
    status("ok", "kpsewhich found")
  else
    status("warn", "kpsewhich not found; class-install checks are unavailable")
  end

  if context.support_dir then
    status("ok", "support dir = " .. context.support_dir)
  end

  if not ok then
    error("doctor checks failed")
  end

  print("Doctor checks passed.")
end

return M
