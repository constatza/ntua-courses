local install = require("ntuabeamer_class.install")
local packaging = require("ntuabeamer_class.packaging")

local M = {}

local function usage()
  print([[
ntuabeamer-class

Usage:
  ntuabeamer-class package
  ntuabeamer-class install
  ntuabeamer-class uninstall
]])
end

function M.main(context)
  local command = context.argv[1]

  if not command or command == "--help" or command == "-h" or command == "help" then
    usage()
    return
  end

  if command == "package" then
    packaging.package_framework(context)
    return
  end

  if command == "install" then
    install.install(context)
    return
  end

  if command == "uninstall" then
    install.uninstall()
    return
  end

  usage()
  error("unknown command '" .. tostring(command) .. "'")
end

return M
