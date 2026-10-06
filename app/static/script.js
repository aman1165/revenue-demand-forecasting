const form =
    document.getElementById("predictionForm");

const predictButton =
    document.getElementById("predictButton");

const predictionResult =
    document.getElementById("predictionResult");

const errorMessage =
    document.getElementById("errorMessage");

const resetButton =
    document.getElementById("resetButton");

const sampleButton =
    document.getElementById("sampleButton");

const sampleSelector =
    document.getElementById("sampleSelector");

const guideButton =
    document.getElementById("guideButton");

const featureGuide =
    document.getElementById("featureGuide");


/* ============================================================
   PRODUCT FAMILY OPTIONS
============================================================ */

const familyOptions = [

    "AUTOMOTIVE",
    "BEAUTY",
    "BEVERAGES",
    "BOOKS",
    "BREAD/BAKERY",
    "CELEBRATION",
    "CLEANING",
    "DAIRY",
    "DELI",
    "EGGS",
    "FROZEN FOODS",
    "GROCERY I",
    "GROCERY II",
    "HARDWARE",
    "HOME AND KITCHEN I",
    "HOME AND KITCHEN II",
    "HOME APPLIANCES",
    "HOME CARE",
    "LADIESWEAR",
    "LAWN AND GARDEN",
    "LINGERIE",
    "LIQUOR,WINE,BEER",
    "MAGAZINES",
    "MEATS",
    "PERSONAL CARE",
    "PET SUPPLIES",
    "PLAYERS AND ELECTRONICS",
    "POULTRY",
    "PREPARED FOODS",
    "PRODUCE",
    "SCHOOL AND OFFICE SUPPLIES",
    "SEAFOOD"

];


/* ============================================================
   SAMPLE INPUT DATA
============================================================ */

let sampleInputs = [];


/* ============================================================
   DEFAULT INPUT
============================================================ */

const defaultInput = {

    store_nbr: 1,

    family: "AUTOMOTIVE",

    onpromotion: 5,

    transactions: 1500,

    year: 2017,

    month: 8,

    day: 1,

    day_of_week: 1,

    day_of_year: 213,

    week_of_year: 31,

    quarter: 3,

    is_weekend: 0,

    is_month_start: 1,

    is_month_end: 0,

    sales_lag_1: 100,

    sales_lag_7: 95,

    sales_lag_14: 90,

    sales_lag_28: 88,

    sales_rolling_mean_7: 96,

    sales_rolling_std_7: 10,

    sales_rolling_mean_14: 94,

    sales_rolling_std_14: 11,

    sales_rolling_mean_28: 92,

    sales_rolling_std_28: 12,

    city: "Quito",

    state: "Pichincha",

    type: "D",

    cluster: 13,

    dcoilwtico: 50,

    holiday_count: 0,

    is_holiday: 0
};


/* ============================================================
   POPULATE PRODUCT FAMILY DROPDOWN
============================================================ */

function populateFamilyOptions() {

    const family =
        document.getElementById("family");


    family.innerHTML = "";


    familyOptions.forEach(
        (familyName) => {

            const option =
                document.createElement("option");


            option.value =
                familyName;


            option.textContent =
                familyName;


            family.appendChild(
                option
            );

        }
    );
}


/* ============================================================
   GET VALUE
============================================================ */

function getValue(id) {

    return document.getElementById(id).value;
}


/* ============================================================
   GET NUMBER VALUE
============================================================ */

function getNumberValue(id) {

    return Number(
        getValue(id)
    );
}


/* ============================================================
   BUILD API PAYLOAD
============================================================ */

function buildPayload() {

    return {

        id: Date.now(),

        store_nbr:
            getNumberValue("store_nbr"),

        family:
            getValue("family"),

        onpromotion:
            getNumberValue("onpromotion"),


        year:
            getNumberValue("year"),

        month:
            getNumberValue("month"),

        day:
            getNumberValue("day"),

        day_of_week:
            getNumberValue("day_of_week"),

        day_of_year:
            getNumberValue("day_of_year"),

        week_of_year:
            getNumberValue("week_of_year"),

        quarter:
            getNumberValue("quarter"),


        is_weekend:
            getNumberValue("is_weekend"),

        is_month_start:
            getNumberValue("is_month_start"),

        is_month_end:
            getNumberValue("is_month_end"),


        sales_lag_1:
            getNumberValue("sales_lag_1"),

        sales_lag_7:
            getNumberValue("sales_lag_7"),

        sales_lag_14:
            getNumberValue("sales_lag_14"),

        sales_lag_28:
            getNumberValue("sales_lag_28"),


        sales_rolling_mean_7:
            getNumberValue(
                "sales_rolling_mean_7"
            ),

        sales_rolling_std_7:
            getNumberValue(
                "sales_rolling_std_7"
            ),


        sales_rolling_mean_14:
            getNumberValue(
                "sales_rolling_mean_14"
            ),

        sales_rolling_std_14:
            getNumberValue(
                "sales_rolling_std_14"
            ),


        sales_rolling_mean_28:
            getNumberValue(
                "sales_rolling_mean_28"
            ),

        sales_rolling_std_28:
            getNumberValue(
                "sales_rolling_std_28"
            ),


        city:
            getValue("city"),

        state:
            getValue("state"),

        type:
            getValue("type"),

        cluster:
            getNumberValue("cluster"),


        transactions:
            getNumberValue("transactions"),

        dcoilwtico:
            getNumberValue("dcoilwtico"),

        holiday_count:
            getNumberValue("holiday_count"),

        is_holiday:
            getNumberValue("is_holiday")

    };
}


/* ============================================================
   FILL FORM
============================================================ */

function fillForm(data) {

    Object.entries(data).forEach(
        ([key, value]) => {

            if (key === "name") {
                return;
            }


            const element =
                document.getElementById(key);


            if (!element) {
                return;
            }


            element.value = value;

        }
    );


    predictionResult.textContent =
        "—";


    errorMessage.textContent =
        "";

}


/* ============================================================
   LOAD SAMPLE JSON
============================================================ */

async function loadSampleInputs() {

    try {

        const response =
            await fetch(
                "/examples/sample_inputs.json"
            );


        if (!response.ok) {

            throw new Error(
                "Unable to load sample inputs."
            );

        }


        sampleInputs =
            await response.json();


        sampleSelector.innerHTML =
            `
            <option value="">
                Select a sample...
            </option>
            `;


        sampleInputs.forEach(
            (sample, index) => {

                const option =
                    document.createElement(
                        "option"
                    );


                option.value =
                    index;


                option.textContent =
                    sample.name;


                sampleSelector.appendChild(
                    option
                );

            }
        );


    } catch (error) {

        errorMessage.textContent =
            error.message;

    }

}


/* ============================================================
   LOAD SELECTED SAMPLE
============================================================ */

sampleButton.addEventListener(
    "click",
    () => {

        const selectedIndex =
            sampleSelector.value;


        if (
            selectedIndex === ""
        ) {

            errorMessage.textContent =
                "Please select a sample input first.";

            return;

        }


        const selectedSample =
            sampleInputs[
                Number(selectedIndex)
            ];


        fillForm(
            selectedSample
        );


        errorMessage.textContent =
            "";


        predictionResult.textContent =
            "—";

    }
);


/* ============================================================
   RESET
============================================================ */

resetButton.addEventListener(
    "click",
    () => {

        fillForm(
            defaultInput
        );


        sampleSelector.value =
            "";


        errorMessage.textContent =
            "";

    }
);


/* ============================================================
   FEATURE GUIDE
============================================================ */

guideButton.addEventListener(
    "click",
    () => {

        featureGuide.classList.toggle(
            "hidden"
        );


        const isHidden =
            featureGuide.classList.contains(
                "hidden"
            );


        guideButton.textContent =
            isHidden
                ? "ⓘ Feature Guide"
                : "ⓘ Hide Guide";

    }
);


/* ============================================================
   PREDICTION
============================================================ */

form.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        errorMessage.textContent =
            "";


        predictionResult.textContent =
            "—";


        predictButton.disabled =
            true;


        predictButton.innerHTML =
            "⏳ Predicting...";


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                buildPayload()
                            )
                    }
                );


            const data =
                await response.json();


            if (!response.ok) {

                throw new Error(
                    data.detail ||
                    "Prediction failed."
                );

            }


            predictionResult.textContent =
                Number(
                    data.predicted_sales
                ).toLocaleString(
                    undefined,
                    {
                        maximumFractionDigits: 2
                    }
                );


        } catch (error) {

            errorMessage.textContent =
                error.message;

        } finally {

            predictButton.disabled =
                false;


            predictButton.innerHTML =
                '✨ Predict Demand <span>›</span>';

        }

    }
);


/* ============================================================
   PAGE INITIALIZATION
============================================================ */

populateFamilyOptions();

fillForm(
    defaultInput
);

loadSampleInputs();