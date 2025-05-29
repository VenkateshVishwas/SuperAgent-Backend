const { spawn } = require("child_process");
const path = require("path");

exports.queryProcess = (req, res) => {
    const query = req.body.query;

    if (!query) {
        return res.status(400).json({ error: "Missing query" });
    }

    const python = process.platform === "win32" ? "python" : "python3";
    const scriptPath = path.join(__dirname, "..", "classify.py");

    console.log("🔥 Query:", query);
    console.log("🛠 Running script:", scriptPath);

    const pythonProcess = spawn(python, [scriptPath, query]);

    let result = "";
    let errorOutput = "";

    pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        errorOutput += data.toString();
    });

    pythonProcess.on("close", (code) => {
        if (errorOutput) {
            console.error("🐍 Python stderr:", errorOutput);
        }

        try {
            console.log("📤 Raw Python output:", result);
            const json = JSON.parse(result);
            return res.status(200).json(json);
        } catch (err) {
            return res.status(500).json({
                error: "❌ Failed to parse Python output",
                raw_output: result,
                stderr: errorOutput,
            });
        }
    });

    pythonProcess.on("error", (err) => {
        return res.status(500).json({
            error: "❌ Failed to start Python process",
            details: err.message,
        });
    });
};
