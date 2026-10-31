local fs = require("ntua_support.fs")
local path = require("ntua_support.path")

local M = {}

local function titleize(name)
  local words = {}
  for word in name:gmatch("[^_-]+") do
    table.insert(words, word:sub(1, 1):upper() .. word:sub(2))
  end
  return table.concat(words, " ")
end

local function starter_path(context, name)
  return path.join(context.support_dir, "templates", "starter", name)
end

function M.create_project(context, target_dir)
  if not target_dir or target_dir == "" then
    error("missing target directory")
  end

  local project_dir = path.abspath(target_dir, context.cwd)
  if fs.exists(project_dir) then
    error("target already exists: " .. project_dir)
  end

  local title = titleize(path.basename(project_dir))
  local main_template = fs.read_file(starter_path(context, "main.tex"))
  main_template = main_template:gsub("%% ntua%-beamer:new%-title\n\\title%b{}", function()
    return "% ntua-beamer:new-title\n\\title{" .. title .. "}"
  end, 1)

  fs.ensure_dir(project_dir)
  fs.ensure_dir(path.join(project_dir, "assets"))
  fs.write_file(path.join(project_dir, "main.tex"), main_template)
  fs.copy_file(starter_path(context, ".latexmkrc"), path.join(project_dir, ".latexmkrc"))
  fs.copy_file(path.join(context.support_dir, "framework", "standalone", "ntuabeamer.cls"), path.join(project_dir, "ntuabeamer.cls"))
  fs.copy_tree(path.join(context.support_dir, "framework", "assets"), path.join(project_dir, "assets"))

  print("Created project -> " .. project_dir)
end

return M
