local M = {}

local is_windows = package.config:sub(1, 1) == "\\"

function M.is_windows()
  return is_windows
end

function M.path_sep()
  return is_windows and ";" or ":"
end

function M.normalize(value)
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

function M.comparable(value)
  value = M.normalize(value)
  if is_windows then
    return value:lower()
  end
  return value
end

function M.is_abs(value)
  value = M.normalize(value)
  return value:match("^/") or value:match("^%a:/")
end

function M.join(...)
  local result = ""

  for index, part in ipairs({...}) do
    if part and part ~= "" then
      part = M.normalize(part)
      if index > 1 then
        part = part:gsub("^/", "")
      end

      if result == "" then
        result = part
      elseif result:sub(-1) == "/" then
        result = result .. part
      else
        result = result .. "/" .. part
      end
    end
  end

  return result
end

function M.dirname(value)
  value = M.normalize(value)

  if value == "" or value == "." then
    return "."
  end

  if value == "/" or value:match("^%a:/$") then
    return value
  end

  return value:match("^(.*)/[^/]+$") or "."
end

function M.basename(value)
  value = M.normalize(value)
  if value == "" or value == "." or value == "/" then
    return value
  end
  return value:match("([^/]+)$") or value
end

function M.stem(value)
  local base = M.basename(value)
  return base:gsub("%.[^%.]+$", "")
end

function M.extension(value)
  local base = M.basename(value)
  return base:match("(%.[^%.]+)$") or ""
end

function M.abspath(value, cwd)
  value = M.normalize(value)
  if value == "" then
    return M.normalize(cwd or "")
  end
  if M.is_abs(value) then
    return value
  end
  return M.normalize(M.join(cwd or "", value))
end

function M.starts_with(parent, child)
  parent = M.comparable(parent)
  child = M.comparable(child)

  if parent == "" then
    return false
  end

  return child == parent or child:sub(1, #parent + 1) == parent .. "/"
end

function M.relpath(target, base)
  target = M.normalize(target)
  base = M.normalize(base)

  if M.starts_with(base, target) then
    local relative = target:sub(#base + 2)
    if relative == "" then
      return "."
    end
    return relative
  end

  return target
end

return M
