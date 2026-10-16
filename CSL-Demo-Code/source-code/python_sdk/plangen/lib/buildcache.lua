-- match(.*(clang).*)
-- Currently we support the following patterns of compilations:
-- .bc -> .elf
-- clang -fno-rtti -O2 -Werror -DFLOAT16_IS_HALF -link-fp16  -mllvm -allocate-symbols=true -target cs -mcpu=einstein <file>.bc -o <file>.elf -fab_w <N> -fab_h <N> -elf_w <N> -elf_h <N> -elf_idx <N>
-- .bc -> .o
-- clang -target cs -mcpu=einstein -O2 -fno-lto -c -o <file>.o <file>.bc
-- .cc -> .bc
-- clang -fno-rtti -O2 -Werror -DFLOAT16_IS_HALF <file>.cc -c -target cs -mcpu=einstein -o <file>.bc --include gather2.nested.tindex/gather2.nested.tindex.h -D<macro1> -D<macro2> ...



require_std("io")
require_std("os")
require_std("string")
require_std("table")
require_std("bcache")


-------------------------------------------------------------------------------
-- Internal helper functions.
-------------------------------------------------------------------------------
local debug = false

local function make_preprocessor_cmd (args, preprocessed_file)
  local preprocess_args = {}

  -- Drop arguments that we do not want/need.
  local drop_next_arg = false
  for i, arg in ipairs(args) do
    local drop_this_arg = drop_next_arg
    drop_next_arg = false
    if arg == "-c" then
      drop_this_arg = true
    elseif arg == "-o" then
      drop_this_arg = true
      drop_next_arg = true
    end
    if not drop_this_arg then
      table.insert(preprocess_args, arg)
    end
  end

  -- Append the required arguments for producing preprocessed output.
  table.insert(preprocess_args, "-E")
  table.insert(preprocess_args, "-P")
  table.insert(preprocess_args, "-o")
  table.insert(preprocess_args, preprocessed_file)

  return preprocess_args
end

local function contains(t, e)
  for i = 1,#t do
    if t[i] == e then return true end
  end
  return false
end

local function is_source_file (path)
  local ext = bcache.get_extension(path):lower()
  return (ext == ".cpp") or (ext == ".cc") or (ext == ".cxx") or (ext == ".c")
end

local function is_plus_arg_opt(arg) 
  local plus_arg_opts = {"-target", "-fab_w", "-fab_h", "-elf_w", "-elf_h", "-elf_idx", "-mllvm"}
  return contains(plus_arg_opts, arg)
end

local function is_plus_file_opt(arg)
  local plus_file_opts = {"-I", "-MF", "-MT", "-MQ", "--include"}
  return contains(plus_file_opts, arg)
end

local function get_input_files_aux(skip_plus_file_opts)
  local input_files = {}
  local skip_next_arg = false
  for i = 2, #ARGS do
    if skip_next_arg then
      skip_next_arg = false
    else
      if (is_plus_arg_opt(ARGS[i]) or (ARGS[i] == "-o")) then
        skip_next_arg = true
      elseif (is_plus_file_opt(ARGS[i])) then
        if (skip_plus_file_opts) then
          skip_next_arg = true
        else
          error("unsupported option for direct mode: " .. ARGS[i])
        end
      elseif (ARGS[i]:sub(1,1) == "-") then
        -- skip
      else
        table.insert(input_files, ARGS[i])
      end
    end
  end
  if (debug) then
    for i, item in ipairs(input_files) do
      print(item)
    end
  end
  return input_files
end

local function read_entire_file(filename)
  local f = assert(io.open(filename, "rb"))
  local contents = f:read("*all")
  f:close()
  return contents
end

-------------------------------------------------------------------------------
-- Wrapper interface implementation.
-------------------------------------------------------------------------------

-- These are outputs of the build
function get_build_files ()
  local files = {}
  local found_object_file = false
  for i = 2, #ARGS do
    local next_idx = i + 1
    if (ARGS[i] == "-o") and (next_idx <= #ARGS) then
      if found_object_file then
        error("Only a single target object file can be specified.")
      end
      files["object"] = ARGS[next_idx]
      if (debug) then
        print(ARGS[next_idx])
      end
      found_object_file = true
    end
  end
  if not found_object_file then
    error("Unable to get the target object file.")
  end
  return files
end

function get_program_id ()
  -- Get the git commit of the compiler.
  local result = bcache.run({ARGS[1], "-v"})
  if result.return_code ~= 0 then
    error("Unable to get the compiler version information string.")
  end
  i, j = string.find(result.std_err, "\n")
  complete_version=string.sub(result.std_err, 1, j-1)
  return complete_version
end

-- These are the inputs of the build
function get_relevant_arguments ()
  local filtered_args = {}
  -- The first argument is the compiler binary without the path.
  table.insert(filtered_args, bcache.get_file_part(ARGS[1]))
  
  -- Note: We always skip the first arg since we have handled it already.
  local skip_next_arg = true
  local relevant_arg = ""
  local input_file_found = false
  for i, arg in ipairs(ARGS) do
    if not skip_next_arg then
      if (append_next_arg) then
        relevant_arg = relevant_arg .. " " .. arg
        table.insert(filtered_args, relevant_arg)
        append_next_arg = false
      elseif (is_plus_file_opt(arg) or (arg == "-o")) then
        skip_next_arg = true
      elseif (arg:sub(1,2) == "-D") then
        -- skip
      elseif (is_plus_arg_opt(arg)) then
        relevant_arg = arg
        append_next_arg = true
      elseif (arg:sub(1,1) ~= "-") then
        -- skip (this is the input file)
        if (input_file_found) then
          error("have seen more than one input file")
        end
        input_file_found = true
      else
        table.insert(filtered_args, arg)
      end
    else
      skip_next_arg = false
    end
  end
  if (not input_file_found) then
    error("didn't find input file")
  end
  table.sort(filtered_args)
  if (debug) then
    print("relevant arguments are: ")
    for i, item in ipairs(filtered_args) do
      print(item)
    end
  end
  return filtered_args
end

function preprocess_source ()
  -- Check if this is a compilation command that we support.

  -- ensure input file is a C or C++ file
  input_files = get_input_files_aux(true)
  if (#input_files > 1) then
    error("multi-input file, preprocessing not supported")    
  end
  for i, item in ipairs(input_files) do
    if not is_source_file(item) then
      if (debug) then
        print("as-is preprocessing")
      end
      return read_entire_file(item)
    end
  end

  local is_object_compilation = false
  local has_object_output = false
  for i, arg in ipairs(ARGS) do
    if arg == "-c" then
      is_object_compilation = true
    elseif arg == "-o" then
      has_object_output = true
    elseif arg:sub(1, 1) == "@" then
      error("Response files are currently not supported.")
    end
  end
  if (not is_object_compilation) or (not has_object_output) then
    if (debug) then
      print("preprocessing not supported")
    end
    return "" -- error("Unsupported complation command.")
  end

  -- Run the preprocessor step.
  local preprocessed_file = os.tmpname()
  local preprocessor_args = make_preprocessor_cmd(ARGS, preprocessed_file)
  local result = bcache.run(preprocessor_args)
  if result.return_code ~= 0 then
    os.remove(preprocessed_file)
    error("Preprocessing command was unsuccessful.")
  end

  -- Read and return the preprocessed file.
  local preprocessed_source = read_entire_file(preprocessed_file)
  os.remove(preprocessed_file)
  return preprocessed_source
end
