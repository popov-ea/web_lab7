const MongoClient = require("mongodb").MongoClient;
const fs = require("fs");
const natural = require("natural");
let tfidf = new natural.TfIdf();
natural.PorterStemmerRu.attach();
const classifierFilePath = "classifier.json";

let mongoClient = new MongoClient('mongodb://localhost:27017', { useNewUrlParser: true, useUnifiedTopology: true});

exports.index = function (request, response) {
    response.render("index");
}

exports.addArticlePage = function (request, response) {
    response.render("addArticle");
}

exports.addArticle = function (request, response) {
    const txt = request.body.articleText;
    if (!fs.existsSync(classifierFilePath)) {
        response.render("noClassifier");
    }
    tfidf = new natural.TfIdf(JSON.parse(fs.readFileSync(classifierFilePath)));
    tfidf.addDocument(txt.tokenizeAndStem());
    const dictionary = getDictionary(tfidf);
    const sportsCentroid = getCentroid(tfidf, "sport");
    const techCentroid = getCentroid(tfidf, "tech");
    const distanceToSport = getDistance(dictionary, sportsCentroid, tfidf);
    const distanceToTech = getDistance(dictionary, techCentroid, tfidf);
    const classificationResult = [
        {category: "Спорт", categoryKey: "sport", distance: distanceToSport},
        {category: "Технологии", categoryKey: "tech", distance: distanceToTech}
    ].sort((a,b) => a.distance - b.distance);
    
    saveArticle(request.body.articleHeader, request.body.articleText, classificationResult[0].categoryKey);

    response.render("classificationResult", {
        classificationResult: classificationResult
    })
}

exports.trainPage = function (request, response) {
    response.render("train");
}

exports.train = function (request, response) {    
    if (fs.existsSync(classifierFilePath)) {
        tfidf = new natural.TfIdf(JSON.parse(fs.readFileSync(classifierFilePath)));
    } else {
        tfidf = new natural.TfIdf();
    }
    const category = request.body.category;
    request.files.forEach(f => {
        tfidf.addDocument(f.buffer.toString().tokenizeAndStem(), category);
    });
    fs.writeFileSync(classifierFilePath, JSON.stringify(tfidf));
    response.redirect("/");
}

exports.getArticlesByCategory = function (request, response) {
    mongoClient.connect()
        .then((c) => 
            c.db("web_lab7")
                .collection("articles")
                .find({
                    category: request.params.category
                })
                .toArray()
        )
        .then((articles) => {
            response.render("articles", {
                articles: (articles || [])
            });
        })
}

function getCentroid(tfidf, category) {
    const dictionary = getDictionary(tfidf);
    const docsCount = getDocsCount(tfidf, category);
    const sumVector = [];

    tfidf.documents.forEach((d, ind) => {
        if (d.__key === category) {
            dictionary.forEach((w,wInd) => {
                const wTfiIdf = tfidf.tfidf(w, ind);
                sumVector[wInd] = (sumVector[wInd] || 0) + wTfiIdf;
            })
        }
    });

    return sumVector.map(v => v/docsCount);
}

function getDictionary(tfidf) {
    let result = new Set();
    tfidf.documents.forEach((d,i) => {
        result = new Set([...result, ...tfidf.listTerms(i).map(t => t.term)]);
    })
    return [...result];
}

function getDocsCount(tfidf, category) {
    return tfidf.documents.filter(d => d.__key === category).length;
}

function getDistance(dictionary, classCentroid, tfidf) {
    let sum = 0;
    classCentroid.forEach((c, i) => {
        sum += Math.pow((c - tfidf.tfidf(dictionary[i], tfidf.documents.length - 1)),2);
    });

    return Math.sqrt(sum);
}

function saveArticle(header, text, category) {
    return mongoClient.connect()
        .then(c => c.db("web_lab7").collection("articles").insertOne({
            header: header,
            text: text,
            category: category
        }));
}