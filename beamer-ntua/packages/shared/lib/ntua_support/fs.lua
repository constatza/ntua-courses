local lfs = require("lfs")

local path = require("ntua_support.path")

local M = {}

function M.exists(target)
  return lfs.attributes(target) ~= nil
end

function M.is_dir(target)
  local attr = lfs.attributes(target)
  return attr and attr.mode == "directory" or false
end

function M.is_file(target)
  local attr = lfs.attributes(target)
  return attr and attr.mode == "file" or false
end

function M.ensure_dir(target)
  target = path.normalize(target)
  if target == "" or target == "." or M.is_dir(target) then
    return
  end

  local prefix = ""
  local rest = target

  if target:match("^%a:/") then
    prefix = target:sub(1, 3)
    rest = target:sub(4)
  elseif target:sub(1, 1) == "/" then
    prefix = "/"
    rest = target:sub(2)
  end

  local current = prefix
  for segment in rest:gmatch("[^/]+") do
    if current == "" or current == "/" then
      current = current .. segment
    else
      current = current .. "/" .. segment
    end

    if not M.exists(current) then
      local ok, err = lfs.mkdir(current)
      if not ok then
        error("failed to create directory " .. current .. ": " .. tostring(err))
      end
    end
  end
end

function M.read_file(target)
  local handle, err = io.open(target, "rb")
  if not handle then
    error("failed to open " .. target .. ": " .. tostring(err))
  end

  local content = handle:read("*a")
  handle:close()
  return content
end

function M.write_file(target, content)
  M.ensure_dir(path.dirname(target))

  local handle, err = io.open(target, "wb")
  if not handle then
    error("failed to open " .. target .. " for writing: " .. tostring(err))
  end

  handle:write(content)
  handle:close()
end

function M.copy_file(src, dst)
  M.write_file(dst, M.read_file(src))
end

function M.copy_tree(src, dst)
  M.ensure_dir(dst)

  local entries = {}
  for entry in lfs.dir(src) do
    if entry ~= "." and entry ~= ".." then
      table.insert(entries, entry)
    end
  end
  table.sort(entries)

  for _, entry in ipairs(entries) do
    local source = path.join(src, entry)
    local target = path.join(dst, entry)
    if M.is_dir(source) then
      M.copy_tree(source, target)
    else
      M.copy_file(source, target)
    end
  end
end

function M.remove_tree(target)
  target = path.normalize(target)
  if not M.exists(target) then
    return
  end

  if M.is_file(target) then
    local ok, err = os.remove(target)
    if not ok then
      error("failed to remove file " .. target .. ": " .. tostring(err))
    end
    return
  end

  local entries = {}
  for entry in lfs.dir(target) do
    if entry ~= "." and entry ~= ".." then
      table.insert(entries, entry)
    end
  end
  table.sort(entries)

  for _, entry in ipairs(entries) do
    M.remove_tree(path.join(target, entry))
  end

  local ok, err = lfs.rmdir(target)
  if not ok then
    error("failed to remove directory " .. target .. ": " .. tostring(err))
  end
end

function M.list_dir_names(target)
  local entries = {}
  for entry in lfs.dir(target) do
    if entry ~= "." and entry ~= ".." then
      table.insert(entries, entry)
    end
  end
  table.sort(entries)
  return entries
end

function M.walk_files(root, visit_file, should_skip_dir)
  local function walk(current)
    local entries = {}
    for entry in lfs.dir(current) do
      if entry ~= "." and entry ~= ".." then
        table.insert(entries, entry)
      end
    end
    table.sort(entries)

    for _, entry in ipairs(entries) do
      local target = path.join(current, entry)
      if M.is_dir(target) then
        if not (should_skip_dir and should_skip_dir(target, entry)) then
          walk(target)
        end
      else
        visit_file(target, entry)
      end
    end
  end

  walk(root)
end

function M.is_writable_dir(target)
  if not M.is_dir(target) then
    return false
  end

  local probe = path.join(target, ".ntua-beamer-write-test-" .. tostring(os.time()) .. "-" .. tostring(math.random(1000, 9999)))
  local handle = io.open(probe, "wb")
  if not handle then
    return false
  end

  handle:write("ok")
  handle:close()
  os.remove(probe)
  return true
end

return M
