local fs = require("ntua_support.fs")
local path = require("ntua_support.path")
local process = require("ntua_support.process")

local M = {}

function M.build_output(source)
  local source_dir = path.dirname(source)
  local stem = path.stem(source)
  local build_dir = path.join(source_dir, ".build")

  return {
    build_dir = build_dir,
    pdf = path.join(build_dir, stem .. ".pdf"),
  }
end

function M.build_source(source)
  if not process.command_exists("latexmk") then
    error("latexmk is required but was not found on PATH")
  end

  local source_dir = path.dirname(source)
  local source_name = path.basename(source)
  local output = M.build_output(source)

  fs.ensure_dir(output.build_dir)

  local success = process.run_command({
    "latexmk",
    "-pdf",
    "-interaction=nonstopmode",
    "-halt-on-error",
    "-auxdir=.build",
    "-outdir=.build",
    source_name,
  }, {
    cwd = source_dir,
  })

  if not success then
    os.exit(1)
  end

  if not fs.exists(output.pdf) then
    error("build completed without producing " .. output.pdf)
  end

  return output
end

return M
