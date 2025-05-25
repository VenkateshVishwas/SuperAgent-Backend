const { spawn } = require("child_process");

exports.queryProcess = (req, res) => {
    const query = req.body.query;

    if (!query) {
        return res.status(400).json({ error: "Missing query" });
    }

    // Correct path to classify.py relative to this controller file
    const pythonProcess = spawn("python3", ["../classify.py", query], {
        cwd: __dirname, // ensure working directory is the controller's directory
    });

    let result = "";

    pythonProcess.stdout.on("data", (data) => {
        result += data.toString();
    });

    pythonProcess.stderr.on("data", (data) => {
        console.error(`Python error: ${data}`);
    });

    pythonProcess.on("close", () => {
        try {
            const output = JSON.parse(result);
            res.json(output);
        } catch (error) {
            res.status(500).json({ error: "Failed to parse model output" });
        }
    });
};
