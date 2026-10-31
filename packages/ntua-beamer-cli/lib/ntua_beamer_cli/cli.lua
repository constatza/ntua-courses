local build = require("ntua_beamer_cli.build")
local doctor = require("ntua_beamer_cli.doctor")
local roots = require("ntua_beamer_cli.roots")
local template = require("ntua_beamer_cli.template")

local fs = require("ntua_support.fs")
local path = require("ntua_support.path")

local M = {}

local function usage()
  print([[
ntua-beamer

Usage:
  ntua-beamer doctor
  ntua-beamer new <directory>
  ntua-beamer build [path]
  ntua-beamer build --all [directory]
  ntua-beamer export <path> --output <file-or-dir>
]])
end

local function parse_build_args(arguments)
  local options = {
    all = false,
    target = nil,
  }

  local index = 2
  while index <= #arguments do
    local value = arguments[index]
    if value == "--all" then
      options.all = true
    elseif not options.target then
      options.target = value
    else
      error("unexpected argument '" .. tostring(value) .. "'")
    end
    index = index + 1
  end

  return options
end

local function parse_export_args(arguments)
  local options = {
    target = nil,
    output = nil,
  }

  local index = 2
  while index <= #arguments do
    local value = arguments[index]
    if value == "--output" then
      index = index + 1
      if index > #arguments then
        error("missing value for --output")
      end
      options.output = arguments[index]
    elseif not options.target then
      options.target = value
    else
      error("unexpected argument '" .. tostring(value) .. "'")
    end
    index = index + 1
  end

  if not options.target then
    error("missing export target")
  end

  if not options.output then
    error("missing --output")
  end

  return options
end

local function resolve_export_target(output, source, cwd)
  local absolute = path.abspath(output, cwd)

  local export_is_dir = fs.exists(absolute) and fs.is_dir(absolute)
  if export_is_dir then
    return path.join(absolute, path.stem(source) .. ".pdf")
  end

  local trailing_sep = output:sub(-1) == "/" or output:sub(-1) == "\\"
  if trailing_sep then
    return path.join(absolute, path.stem(source) .. ".pdf")
  end

  if path.extension(absolute) == ".pdf" then
    return absolute
  end

  return path.join(absolute, path.stem(source) .. ".pdf")
end

local function run_build(context, arguments)
  local options = parse_build_args(arguments)

  if options.all then
    for _, source in ipairs(roots.resolve_all(options.target, context.cwd)) do
      print("Building " .. source)
      local output = build.build_source(source)
      print("Built " .. source .. " -> " .. output.pdf)
    end
    return
  end

  local source = roots.resolve_one(options.target, context.cwd)
  local output = build.build_source(source)
  print("Built " .. source .. " -> " .. output.pdf)
end

local function run_export(context, arguments)
  local options = parse_export_args(arguments)
  local source = roots.resolve_one(options.target, context.cwd)
  local build_output = build.build_source(source)
  local export_target = resolve_export_target(options.output, source, context.cwd)

  fs.ensure_dir(path.dirname(export_target))
  fs.copy_file(build_output.pdf, export_target)
  print("Exported " .. source .. " -> " .. export_target)
end

function M.main(context)
  local command = context.argv[1]

  if not command or command == "--help" or command == "-h" or command == "help" then
    usage()
    return
  end

  if command == "doctor" then
    doctor.run(context)
    return
  end

  if command == "new" then
    template.create_project(context, context.argv[2])
    return
  end

  if command == "build" then
    run_build(context, context.argv)
    return
  end

  if command == "export" then
    run_export(context, context.argv)
    return
  end

  usage()
  error("unknown command '" .. tostring(command) .. "'")
end

return M
