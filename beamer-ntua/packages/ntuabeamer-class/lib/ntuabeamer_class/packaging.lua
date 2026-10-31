local fs = require("ntua_support.fs")
local path = require("ntua_support.path")

local M = {}

local function trim(value)
  return (value or ""):gsub("^%s+", ""):gsub("%s+$", "")
end

local function filtered_lines(target, skip_first, removals)
  local output = {}
  local count = 0

  for line in (fs.read_file(target) .. "\n"):gmatch("(.-)\n") do
    count = count + 1
    if count > skip_first and not removals[trim(line)] then
      table.insert(output, line)
    end
  end

  while #output > 0 and output[#output] == "" do
    table.remove(output)
  end

  return table.concat(output, "\n")
end

function M.build_standalone_class(framework_dir)
  local cls_body = filtered_lines(path.join(framework_dir, "ntuabeamer.cls"), 2, {
    ["\\RequirePackage{ntuabeamer}"] = true,
  })
  local sty_body = filtered_lines(path.join(framework_dir, "ntuabeamer.sty"), 2, {
    ["\\input{macros.tex}"] = true,
    ["\\input{theme.tex}"] = true,
  })
  local macros_body = trim(fs.read_file(path.join(framework_dir, "macros.tex")))
  local theme_body = trim(fs.read_file(path.join(framework_dir, "theme.tex")))

  return table.concat({
    "\\NeedsTeXFormat{LaTeX2e}",
    "\\ProvidesClass{ntuabeamer}[2026/03/24 NTUA Beamer standalone class]",
    cls_body,
    "",
    "%% Inlined from ntuabeamer.sty",
    sty_body,
    "",
    "%% Inlined from macros.tex",
    macros_body,
    "",
    "%% Inlined from theme.tex",
    theme_body,
    "",
  }, "\n")
end

function M.package_framework(context, output_root)
  local framework_dir = context.framework_dir
  local assets_dir = path.join(framework_dir, "assets")
  local dist_dir = output_root and path.abspath(output_root, context.cwd) or path.join(context.cwd, "dist")
  local modular_dir = path.join(dist_dir, "ntuabeamer-modular")
  local standalone_dir = path.join(dist_dir, "ntuabeamer-standalone")

  fs.remove_tree(modular_dir)
  fs.remove_tree(standalone_dir)
  fs.ensure_dir(modular_dir)
  fs.ensure_dir(standalone_dir)

  for _, filename in ipairs({"ntuabeamer.cls", "ntuabeamer.sty", "macros.tex", "theme.tex"}) do
    fs.copy_file(path.join(framework_dir, filename), path.join(modular_dir, filename))
  end
  fs.copy_tree(assets_dir, modular_dir)

  fs.write_file(path.join(standalone_dir, "ntuabeamer.cls"), M.build_standalone_class(framework_dir))
  print("Packaged framework -> " .. dist_dir)
end

return M
