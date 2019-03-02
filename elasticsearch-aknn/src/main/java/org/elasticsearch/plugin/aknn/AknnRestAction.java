/*
 * Copyright [2018] [Alex Klibisz]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
package org.elasticsearch.plugin.aknn;

import org.apache.commons.math3.util.Pair;
import org.elasticsearch.action.bulk.BulkRequestBuilder;
import org.elasticsearch.action.bulk.BulkResponse;
import org.elasticsearch.action.get.GetResponse;
import org.elasticsearch.action.index.IndexResponse;
import org.elasticsearch.action.search.SearchResponse;
import org.elasticsearch.client.node.NodeClient;
import org.elasticsearch.common.StopWatch;
import org.elasticsearch.common.inject.Inject;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.xcontent.XContentBuilder;
import org.elasticsearch.common.xcontent.XContentHelper;
import org.elasticsearch.common.xcontent.XContentParser;
import org.elasticsearch.index.query.BoolQueryBuilder;
import org.elasticsearch.index.query.QueryBuilder;
import org.elasticsearch.index.query.QueryBuilders;
import org.elasticsearch.rest.BaseRestHandler;
import org.elasticsearch.rest.BytesRestResponse;
import org.elasticsearch.rest.RestController;
import org.elasticsearch.rest.RestRequest;
import org.elasticsearch.rest.RestStatus;
import org.elasticsearch.search.SearchHit;

import java.io.IOException;
import java.util.Collections;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static java.lang.Math.min;
import static org.elasticsearch.rest.RestRequest.Method.GET;
import static org.elasticsearch.rest.RestRequest.Method.POST;

import org.elasticsearch.common.xcontent.DeprecationHandler;

public class AknnRestAction extends BaseRestHandler {

    public static String NAME = "_aknn";
    private final String NAME_SEARCH = "_aknn_search";
    private final String NAME_INDEX = "_aknn_index";
    private final String NAME_CREATE = "_aknn_create";

    // TODO: check how parameters should be defined at the plugin level.
    private final String HASHES_KEY = "_aknn_hashes";
    private final String VECTOR_KEY = "_aknn_vector";
    private final Integer K1_DEFAULT = 99;
    private final Integer K2_DEFAULT = 10;

    // TODO: add an option to the index endpoint handler that empties the cache.
    private Map<String, LshModel> lshModelCache = new HashMap<>();

    @Inject
    public AknnRestAction(Settings settings, RestController controller) {
        super(settings);
        controller.registerHandler(GET, "/{index}/{type}/{id}/" + NAME_SEARCH, this);
        controller.registerHandler(POST, NAME_INDEX, this);
        controller.registerHandler(POST, NAME_CREATE, this);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest restRequest, NodeClient client) throws IOException {
        if (restRequest.path().endsWith(NAME_SEARCH))
            return handleSearchRequest(restRequest, client);
        else if (restRequest.path().endsWith(NAME_INDEX))
            return handleIndexRequest(restRequest, client);
        else
            return handleCreateRequest(restRequest, client);
    }

    private static Double euclideanDistance(List<Double> A, List<Double> B) {
        Double squaredDistance = 0.;
        for (int i = 0; i < A.size(); i++)
            squaredDistance += Math.pow(A.get(i) - B.get(i), 2);
        return Math.sqrt(squaredDistance);
    }

    private static Double euclideanDistances(List<List<Double>> A, List<List<Double>> B) {
        List<Double> eDistances = new ArrayList<Double>();

        for (List<Double> A_frame : A) {
            for (List<Double> B_frame : B) {
                eDistances.add(euclideanDistance(A_frame, B_frame));
            }
        }

        // eDistances.sort(Collections.reverseOrder());
        Collections.sort(eDistances);

        int top = A.size();
        if(B.size() < A.size()){
            top = B.size();
        }
        eDistances = eDistances.subList(0, top);

        Double eDistancesSum = 0.;
        for (Double sd : eDistances) {
            eDistancesSum += sd;
        }
        return eDistancesSum / eDistances.size();
    }

    private static Double cosineDistance(List<Double> A, List<Double> B) {
        Double dotProduct = 0.0;
        Double magnitude1 = 0.0;
        Double magnitude2 = 0.0;
        Double cosineSimilarity = 0.0;

        for (int i = 0; i < A.size(); i++) {
            dotProduct += A.get(i) * B.get(i);  //a.b
            magnitude1 += Math.pow(A.get(i), 2);  //(a^2)
            magnitude2 += Math.pow(B.get(i), 2); //(b^2)
        }

        magnitude1 = Math.sqrt(magnitude1);//sqrt(a^2)
        magnitude2 = Math.sqrt(magnitude2);//sqrt(b^2)

        if (magnitude1 != 0.0 | magnitude2 != 0.0) {
            cosineSimilarity = dotProduct / (magnitude1 * magnitude2);
        } else {
            return 0.0;
        }
        return cosineSimilarity;
    }

    private static Double cosineDistances(List<List<Double>> A, List<List<Double>> B) {
        List<Double> cDistances = new ArrayList<Double>();

        for (List<Double> A_frame : A) {
            for (List<Double> B_frame : B) {
                cDistances.add(cosineDistance(A_frame, B_frame));
            }
        }

        Collections.sort(cDistances);

        int top = A.size();
        if(B.size() < A.size()){
            top = B.size();
        }
        cDistances = cDistances.subList(0, top);

        Double cDistancesSum = 0.;
        for (Double sd : cDistances) {
            cDistancesSum += sd;
        }
        return cDistancesSum / cDistances.size();
    }

    private static Pair<Float, List<List<Double>>> reduceVectors(List<List<Double>> vectorList) {
        List<List<Double>> newVectorList = new ArrayList<>();
        newVectorList.add(vectorList.get(0));

        // loop over all frames except last one
        for (Integer f1 = 0; f1 < vectorList.size(); f1++){
            Double eDistance1 = 0.;
            boolean FirstLoop = true;

            // loop over frames staring from frame previous loop + 1
            for (Integer f2 = 0; f2 < newVectorList.size(); f2++){
                Double eDistance2 = euclideanDistance(vectorList.get(f1), newVectorList.get(f2));

                if (FirstLoop){
                    eDistance1 = eDistance2;
                    FirstLoop = false;
                }else{
                    if (eDistance2 < eDistance1){
                        eDistance1 = eDistance2;
                    }
                }
            }
            // keep vectors of frames that are not the same
            if (eDistance1 > 5){
                newVectorList.add(vectorList.get(f1));
            }
        }
        Float Diff = (100 / (float)vectorList.size() ) * (float)newVectorList.size();

        return new Pair<Float, List<List<Double>>>(Diff, newVectorList);
    }

    private RestChannelConsumer handleSearchRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to Time Search Request");

        // Parse request parameters.
        stopWatch.start("Parse request parameters");
        final String index = restRequest.param("index");
        final String type = restRequest.param("type");
        final String id = restRequest.param("id");
        final Integer k1 = restRequest.paramAsInt("k1", K1_DEFAULT);
        final Integer k2 = restRequest.paramAsInt("k2", K2_DEFAULT);
        stopWatch.stop();

        logger.info("Get query document at {}/{}/{}", index, type, id);
        stopWatch.start("Get query document");
        GetResponse queryGetResponse = client.prepareGet(index, type, id).get();
        Map<String, Object> baseSource = queryGetResponse.getSource();
        stopWatch.stop();

        logger.info("Parse query document hashes");
        stopWatch.start("Parse query document hashes");
        @SuppressWarnings("unchecked")
        List<Map<String, Long>> queryHashes = (List<Map<String, Long>>) baseSource.get(HASHES_KEY);
        stopWatch.stop();

        stopWatch.start("Parse query document vector");
        @SuppressWarnings("unchecked")
        List<List<Double>> queryVector = (List<List<Double>>) baseSource.get(VECTOR_KEY);
        stopWatch.stop();

        // Retrieve the documents with most matching hashes. https://stackoverflow.com/questions/10773581
        // FIXME fails with too_many_clauses error when searching for gif/video
        logger.info("Build boolean query from hashes");
        stopWatch.start("Build boolean query from hashes");
        QueryBuilder queryBuilder = QueryBuilders.boolQuery();
        for (Map<String, Long> frame : queryHashes) {
            for (Map.Entry<String, Long> entry : frame.entrySet()) {
                String termKey = HASHES_KEY + "." + entry.getKey();
                List<Long> termValues = new ArrayList<>();
                for (Map<String, Long> f2 : queryHashes) {
                    Object termValue = (Object)f2.get(entry.getKey());
                    termValues.add(Long.valueOf((Integer)termValue));
                }
                ((BoolQueryBuilder) queryBuilder).should(QueryBuilders.termsQuery(termKey, termValues));
            }
        }
        stopWatch.stop();

        logger.info("Execute boolean search");
        stopWatch.start("Execute boolean search");
        SearchResponse approximateSearchResponse = client
                .prepareSearch(index)
                .setTypes(type)
                .setFetchSource("*", HASHES_KEY)
                .setQuery(queryBuilder)
                .setSize(k1)
                .get();
        stopWatch.stop();

        // Compute exact KNN on the approximate neighbors.
        // Recreate the SearchHit structure, but remove the vector and hashes.
        logger.info("Compute exact distance and construct search hits");
        stopWatch.start("Compute exact distance and construct search hits");
        List<Map<String, Object>> modifiedSortedHits = new ArrayList<>();
        for (SearchHit hit: approximateSearchResponse.getHits()) {
            Map<String, Object> hitSource = hit.getSourceAsMap();
            @SuppressWarnings("unchecked")
            List<List<Double>> hitVector = (List<List<Double>>) hitSource.get(VECTOR_KEY);
            hitSource.remove(VECTOR_KEY);
            hitSource.remove(HASHES_KEY);
            modifiedSortedHits.add(new HashMap<String, Object>() {{
                put("_index", hit.getIndex());
                put("_id", hit.getId());
                put("_type", hit.getType());
                put("_score", euclideanDistances(queryVector, hitVector));
                put("_source", hitSource);
            }});
        }
        stopWatch.stop();

        logger.info("Sort search hits by exact distance");
        stopWatch.start("Sort search hits by exact distance");
        modifiedSortedHits.sort(Comparator.comparingDouble(x -> (Double) x.get("_score")));
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.field("timed_out", false);
            builder.startObject("hits");
            builder.field("max_score", 0);

            // In some cases there will not be enough approximate matches to return *k2* hits. For example, this could
            // be the case if the number of bits per table in the LSH model is too high, over-partioning the space.
            builder.field("total", min(k2, modifiedSortedHits.size()));
            builder.field("hits", modifiedSortedHits.subList(0, min(k2, modifiedSortedHits.size())));
            builder.endObject();
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleCreateRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time create request");
        logger.info("Parse request");
        stopWatch.start("Parse request");

        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(),
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                restRequest.content(),
                restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        @SuppressWarnings("unchecked")
        Map<String, Object> sourceMap = (Map<String, Object>) contentMap.get("_source");

        final String _index = (String) contentMap.get("_index");
        final String _type = (String) contentMap.get("_type");
        final String _id = (String) contentMap.get("_id");
        final String description = (String) sourceMap.get("_aknn_description");
        final Integer nbTables = (Integer) sourceMap.get("_aknn_nb_tables");
        final Integer nbBitsPerTable = (Integer) sourceMap.get("_aknn_nb_bits_per_table");
        final Integer nbDimensions = (Integer) sourceMap.get("_aknn_nb_dimensions");
        @SuppressWarnings("unchecked")
        final List<List<Double>> vectorSample = (List<List<Double>>) contentMap.get("_aknn_vector_sample");
        stopWatch.stop();

        logger.info("Fit LSH model from sample vectors");
        stopWatch.start("Fit LSH model from sample vectors");
        LshModel lshModel = new LshModel(nbTables, nbBitsPerTable, nbDimensions, description);
        lshModel.fitFromVectorSample(vectorSample);
        stopWatch.stop();

        logger.info("Serialize LSH model");
        stopWatch.start("Serialize LSH model");
        Map<String, Object> lshSerialized = lshModel.toMap();
        stopWatch.stop();

        logger.info("Index LSH model");
        stopWatch.start("Index LSH model");
        IndexResponse indexResponse = client.prepareIndex(_index, _type, _id)
                .setSource(lshSerialized)
                .get();
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }

    private RestChannelConsumer handleIndexRequest(RestRequest restRequest, NodeClient client) throws IOException {

        StopWatch stopWatch = new StopWatch("StopWatch to time bulk indexing request");

        logger.info("Parse request parameters");
        stopWatch.start("Parse request parameters");
        XContentParser xContentParser = XContentHelper.createParser(
                restRequest.getXContentRegistry(),
                DeprecationHandler.THROW_UNSUPPORTED_OPERATION,
                restRequest.content(),
                restRequest.getXContentType());
        Map<String, Object> contentMap = xContentParser.mapOrdered();
        final String index = (String) contentMap.get("_index");
        final String type = (String) contentMap.get("_type");
        final String aknnURI = (String) contentMap.get("_aknn_uri");
        @SuppressWarnings("unchecked")
        final List<Map<String, Object>> docs = (List<Map<String, Object>>) contentMap.get("_aknn_docs");
        logger.info("Received {} docs for indexing", docs.size());
        stopWatch.stop();

        // TODO: check if the index exists. If not, create a mapping which does not index continuous values.
        // This is rather low priority, as I tried it via Python and it doesn't make much difference.

        // Check if the LshModel has been cached. If not, retrieve the Aknn document and use it to populate the model.
        LshModel lshModel;
        if (! lshModelCache.containsKey(aknnURI)) {

            // Get the Aknn document.
            logger.info("Get Aknn model document from {}", aknnURI);
            stopWatch.start("Get Aknn model document");
            String[] annURITokens = aknnURI.split("/");
            GetResponse aknnGetResponse = client.prepareGet(annURITokens[0], annURITokens[1], annURITokens[2]).get();
            stopWatch.stop();

            // Instantiate LSH from the source map.
            logger.info("Parse Aknn model document");
            stopWatch.start("Parse Aknn model document");
            lshModel = LshModel.fromMap(aknnGetResponse.getSourceAsMap());
            stopWatch.stop();

            // Save for later.
            lshModelCache.put(aknnURI, lshModel);

        } else {
            logger.info("Get Aknn model document from local cache");
            stopWatch.start("Get Aknn model document from local cache");
            lshModel = lshModelCache.get(aknnURI);
            stopWatch.stop();
        }

        // Prepare documents for batch indexing.
        logger.info("Hash documents for indexing");
        stopWatch.start("Hash documents for indexing");
        BulkRequestBuilder bulkIndexRequest = client.prepareBulk();
        for (Map<String, Object> doc: docs) {
            @SuppressWarnings("unchecked")
            Map<String, Object> source = (Map<String, Object>) doc.get("_source");
            @SuppressWarnings("unchecked")
            List<List<Double>> vectors = (List<List<Double>>) source.get(VECTOR_KEY);

            if(vectors.size() > 1) {    // more than 1 frame (gif/video)
                boolean reduced = false;

                if (source.containsKey("reduced")) {
                    reduced = (boolean) source.get("reduced");  // frames may be reduced externaly
                }
                if (!source.containsKey("frames")) {
                    source.put("frames", vectors.size());
                }

                if (!reduced) {
                    Pair<Float, List<List<Double>>> p = reduceVectors(vectors);
                    reduced = true;
                    vectors = p.getValue();

                    source.put("reduction", p.getKey());
                    source.put(VECTOR_KEY, vectors);
                }
                source.put("reduced", reduced);
            }

            source.put(HASHES_KEY, lshModel.getVectorsHashes(vectors));
            bulkIndexRequest.add(client
                    .prepareIndex(index, type, (String) doc.get("_id"))
                    .setSource(source));
        }
        stopWatch.stop();

        logger.info("Execute bulk indexing");
        stopWatch.start("Execute bulk indexing");
        BulkResponse bulkIndexResponse = bulkIndexRequest.get();
        stopWatch.stop();

        logger.info("Timing summary\n {}", stopWatch.prettyPrint());

        if (bulkIndexResponse.hasFailures()) {
            logger.error("Indexing failed with message: {}", bulkIndexResponse.buildFailureMessage());
            return channel -> {
                XContentBuilder builder = channel.newBuilder();
                builder.startObject();
                builder.field("took", stopWatch.totalTime().getMillis());
                builder.field("error", bulkIndexResponse.buildFailureMessage());
                builder.endObject();
                channel.sendResponse(new BytesRestResponse(RestStatus.INTERNAL_SERVER_ERROR, builder));
            };
        }

        logger.info("Indexed {} docs successfully", docs.size());
        return channel -> {
            XContentBuilder builder = channel.newBuilder();
            builder.startObject();
            builder.field("size", docs.size());
            builder.field("took", stopWatch.totalTime().getMillis());
            builder.endObject();
            channel.sendResponse(new BytesRestResponse(RestStatus.OK, builder));
        };
    }
}
