local fs = require("ntua_support.fs")
local path = require("ntua_support.path")
local process = require("ntua_support.process")

local M = {}

local function get_texmfhome()
  if not process.command_exists("kpsewhich") then
    error("kpsewhich is required but was not found on PATH")
  end

  local output, ok = process.capture_command({"kpsewhich", "-var-value=TEXMFHOME"})
  if not ok or output == "" then
    error("unable to resolve TEXMFHOME via kpsewhich")
  end

  return path.normalize(output)
end

local function refresh_filename_db()
  if process.command_exists("mktexlsr") then
    process.run_command({"mktexlsr"}, {quiet = true})
    return "mktexlsr"
  end

  if process.command_exists("initexmf") then
    process.run_command({"initexmf", "--update-fndb"}, {quiet = true})
    return "initexmf --update-fndb"
  end

  return nil
end

function M.install(context)
  local texmfhome = get_texmfhome()
  local install_dir = path.join(texmfhome, "tex", "latex", "ntuabeamer")

  fs.remove_tree(install_dir)
  fs.ensure_dir(install_dir)

  for _, filename in ipairs({"ntuabeamer.cls", "ntuabeamer.sty", "macros.tex", "theme.tex"}) do
    fs.copy_file(path.join(context.framework_dir, filename), path.join(install_dir, filename))
  end
  fs.copy_tree(path.join(context.framework_dir, "assets"), install_dir)

  local refresh = refresh_filename_db()

  print("Installed class -> " .. install_dir)
  if refresh then
    print("Refreshed TeX filename database with " .. refresh)
  else
    print("No TeX filename refresh command was found.")
  end
end

function M.uninstall()
  local texmfhome = get_texmfhome()
  local install_dir = path.join(texmfhome, "tex", "latex", "ntuabeamer")

  fs.remove_tree(install_dir)
  print("Removed class -> " .. install_dir)

  local refresh = refresh_filename_db()
  if refresh then
    print("Refreshed TeX filename database with " .. refresh)
  else
    print("No TeX filename refresh command was found.")
  end
end

return M
