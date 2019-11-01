using System;
using System.Collections.Generic;

using System.Globalization;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace NeuralNetworkLib
{
    static class JsonLoader
    {
        public static Matrix loadMatrix(JsonMat matJ)
        {
            int r = matJ.Rows;
            int c = matJ.Columns;

            Matrix m = new Matrix(r, c);
            m.values = matJ.Values;
            return m;
        }
    }

    public partial class JsonMat
    {
        [JsonProperty("rows")]
        public int Rows { get; set; }

        [JsonProperty("columns")]
        public int Columns { get; set; }

        [JsonProperty("values")]
        public float[,] Values { get; set; }
    }

    public partial class JsonNN
    {
        [JsonProperty("weights")]
        public JsonMat[] Weights { get; set; }

        [JsonProperty("biases")]
        public JsonMat[] Biases { get; set; }

        [JsonProperty("Vweight")]
        public JsonMat[] Vweight { get; set; }

        [JsonProperty("Vbias")]
        public JsonMat[] Vbias { get; set; }
    }

    public partial class JsonNN
    {
        public static JsonNN FromJson(string json) => JsonConvert.DeserializeObject<JsonNN>(json, Converter.Settings);
    }

    public static class Serialize
    {
        public static string ToJson(this JsonNN self) => JsonConvert.SerializeObject(self, Converter.Settings);
    }

    internal static class Converter
    {
        public static readonly JsonSerializerSettings Settings = new JsonSerializerSettings
        {
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore,
            DateParseHandling = DateParseHandling.None,
            Converters =
            {
                new IsoDateTimeConverter { DateTimeStyles = DateTimeStyles.AssumeUniversal }
            },
        };
    }
}
