const express = require("express");
const appController = require("../controllers/appController");
const bodyParser = require("body-parser");
const multer = require("multer");
var upload = multer(multer.memoryStorage());

const router = express.Router();
const urlencodedParser = bodyParser.urlencoded({extended: false});


router.get("/", appController.index);

router.get("/addArticle", appController.addArticlePage);
router.post("/addArticle", urlencodedParser, appController.addArticle);

router.get("/articles/:category", appController.getArticlesByCategory);

router.get("/train", appController.trainPage);
router.post("/train", upload.array("files"), appController.train);

module.exports = router;