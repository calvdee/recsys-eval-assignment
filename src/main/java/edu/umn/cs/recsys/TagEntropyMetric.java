package edu.umn.cs.recsys;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.sun.org.apache.xerces.internal.util.SynchronizedSymbolTable;
import edu.umn.cs.recsys.dao.ItemTagDAO;
import edu.umn.cs.recsys.dao.ItemTitleDAO;
import it.unimi.dsi.fastutil.longs.Long2LongOpenHashMap;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongArraySet;
import it.unimi.dsi.fastutil.longs.LongSet;
import org.grouplens.lenskit.core.LenskitRecommender;
import org.grouplens.lenskit.eval.algorithm.AlgorithmInstance;
import org.grouplens.lenskit.eval.data.traintest.TTDataSet;
import org.grouplens.lenskit.eval.metrics.AbstractTestUserMetric;
import org.grouplens.lenskit.eval.metrics.TestUserMetricAccumulator;
import org.grouplens.lenskit.eval.metrics.topn.ItemSelectors;
import org.grouplens.lenskit.eval.traintest.TestUser;
import org.grouplens.lenskit.scored.ScoredId;
import org.grouplens.lenskit.vectors.MutableSparseVector;
import org.grouplens.lenskit.vectors.SparseVector;
import org.grouplens.lenskit.vectors.VectorEntry;

import javax.annotation.Nonnull;
import java.util.List;
import java.util.Set;

/**
 * A metric that measures the tag entropy of the recommended items.
 * @author <a href="http://www.grouplens.org">GroupLens Research</a>
 */
public class TagEntropyMetric extends AbstractTestUserMetric {
    private final int listSize;
    private final List<String> columns;

    /**
     * Construct a new tag entropy metric.
     * 
     * @param nitems The number of items to request.
     */
    public TagEntropyMetric(int nitems) {
        listSize = nitems;
        // initialize column labels with list length
        columns = ImmutableList.of(String.format("TagEntropy@%d", nitems));
    }

    /**
     * Make a metric accumulator.  Metrics operate with <em>accumulators</em>, which are created
     * for each algorithm and data set.  The accumulator measures each user's output, and
     * accumulates the results into a global statistic for the whole evaluation.
     *
     * @param algorithm The algorithm being tested.
     * @param data The data set being tested with.
     * @return An accumulator for analyzing this algorithm and data set.
     */
    @Override
    public TestUserMetricAccumulator makeAccumulator(AlgorithmInstance algorithm, TTDataSet data) {
        return new TagEntropyAccumulator();
    }

    /**
     * Return the labels for the (global) columns returned by this metric.
     * @return The labels for the global columns.
     */
    @Override
    public List<String> getColumnLabels() {
        return columns;
    }

    /**
     * Return the lables for the per-user columns returned by this metric.
     */
    @Override
    public List<String> getUserColumnLabels() {
        // per-user and global have the same fields, they just differ in aggregation.
        return columns;
    }


    private class TagEntropyAccumulator implements TestUserMetricAccumulator {
        private double totalEntropy = 0;
        private int userCount = 0;

        /**
         * Evaluate a single test user's recommendations or predictions.
         * @param testUser The user's recommendation result.
         * @return The values for the per-user columns.
         */
        @Nonnull
        @Override
        public Object[] evaluate(TestUser testUser) {
            List<ScoredId> recommendations =
                    testUser.getRecommendations(listSize,
                            ItemSelectors.allItems(),
                            ItemSelectors.trainingItems());
            if (recommendations == null) {
                return new Object[1];
            }
            LenskitRecommender lkrec = (LenskitRecommender) testUser.getRecommender();
            ItemTagDAO tagDAO = lkrec.get(ItemTagDAO.class);
            TagVocabulary vocab = lkrec.get(TagVocabulary.class);

            double entropy = 0;

            // Set of movies in case there's duplicates
            LongSet movies = new LongArraySet();
            for(ScoredId score : recommendations) {
                long m = score.getId();
                if(movies.contains(m)) {
                    continue;
                }
                movies.add(m);
            }

            // Vector to sum entropy per-tag
            MutableSparseVector entropyVec = vocab.newTagVector();
            entropyVec.fill(0.0);
            // Recommendation list size
            double listSize = movies.size();
            // Set of all the seen tags
            Set<String> seen = Sets.newHashSet();
            for(long m : movies) {
                // Tags for this movie
                Set<String> tags = Sets.newHashSet(tagDAO.getItemTags(m));
                // Number of tags for this movie
                double size = tags.size();

                MutableSparseVector work = vocab.newTagVector();

                for(String tag : tags) {
                    if(!vocab.hasTag(tag) || seen.contains(tag)) {
                        continue;
                    }

                    seen.add(tag);

                    double p = (1/size) * (1/listSize);
                    long tagId = vocab.getTagId(tag);

                    work.set(tagId, -p * (Math.log(p) / Math.log(2)));
                }

                entropyVec.add(work);
            }

            entropy = entropyVec.sum();

            totalEntropy += entropy;
            userCount += 1;
            return new Object[]{entropy};
        }

        /**
         * Get the final aggregate results.  This is called after all users have been evaluated, and
         * returns the values for the columns in the global output.
         *
         * @return The final, aggregated columns.
         */
        @Nonnull
        @Override
        public Object[] finalResults() {
            // return a single field, the average entropy
            return new Object[]{totalEntropy / userCount};
        }
    }
}
