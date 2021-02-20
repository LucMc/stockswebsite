        $(document).ready(function(){
        graphs = ["MACD", "RSI", "graph", "returns", "ARIMA"]
        giveSwitchesOnclick(graphs);

        graphs.forEach(graph => {
            if ($("#" + graph + "checkbox").is(":checked") == false){
            $("#" + graph).hide();
            }
            else {
            $("#" + graph).show();
            }
        });
        function giveSwitchesOnclick(graphs){
            graphs.forEach(graph => {
                $("#" + graph + "checkbox").click(function () {
                    if ($("#" + graph + "checkbox").is(":checked") == false) {
                        $("#" + graph).hide();
                    }
                    else {
                        $("#" + graph).show();
                    }

                });
              });
            }
            });