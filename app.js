const express = require("express");
const bodyParser = require("body-parser")
const queryRoutes = require("./routes/queryRoutes")


const app = express();

app.use(express.json());
app.use(bodyParser.urlencoded({ extended: true }));

app.use('/api/v1/',queryRoutes);

app.listen(4000, function() {
  console.log("Server started on port 4000");
});