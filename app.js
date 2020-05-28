const express = require("express");
const router = require("./routers/appRouter");
const expressHbs = require("express-handlebars");
const hbs = require("hbs");

const app = express();

app.engine("hbs", expressHbs({
    layoutsDir: "views/layouts",
    defaultLayout: "layout",
    extname: "hbs"
}));
app.set("view engine", "hbs");

app.use("/", router);

app.listen(3000);